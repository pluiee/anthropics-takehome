"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.

# Updates
- 147734: Baseline
- 21794: Use vector instructions
- 17954: Load and store only once
- 17442: Use multiply_add
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def alloc_hash_values(self):
        slots = []

        for (op1, val1, op2, op3, val3) in HASH_STAGES:
            val1_const = self.scratch_const(val1)
            val3_const = self.scratch_const(val3)
            val1_v = self.alloc_scratch(f"{val1}_v", VLEN)
            val3_v = self.alloc_scratch(f"{val3}_v", VLEN)
            slots.append(("valu", ("vbroadcast", val1_v, val1_const)))
            slots.append(("valu", ("vbroadcast", val3_v, val3_const)))
        
        return slots


    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            val1_v = self.scratch[f"{val1}_v"]
            val3_v = self.scratch[f"{val3}_v"]
            slots.append(("valu", (op1, tmp1, val_hash_addr, val1_v)))
            slots.append(("valu", (op3, tmp2, val_hash_addr, val3_v)))
            slots.append(("valu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("vcompare", val_hash_addr, [(round, i+j, "hash_stage", hi) for j in range(VLEN)])))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Scalar implementation using only scalar ALU and load/store.
        """
        # Scratch space addresses
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)

        # Initialize consts at once to make them adjacent in scratch space.
        for i in range(batch_size):
            self.scratch_const(i)

        for i, v in enumerate(init_vars):
            self.add("load", ("load", self.scratch[v], self.scratch_const(i)))
        
        # Broadcast values and constants for vector instructions.
        inp_indices_p_v = self.alloc_scratch("inp_indices_p_v", VLEN)
        self.add("valu", ("vbroadcast", inp_indices_p_v, self.scratch["inp_indices_p"]))
        inp_values_p_v = self.alloc_scratch("inp_values_p_v", VLEN)
        self.add("valu", ("vbroadcast", inp_values_p_v, self.scratch["inp_values_p"]))
        forest_values_p_v = self.alloc_scratch("forest_values_p_v", VLEN)
        self.add("valu", ("vbroadcast", forest_values_p_v, self.scratch["forest_values_p"]))
        n_nodes_v = self.alloc_scratch("n_nodes_v", VLEN)
        self.add("valu", ("vbroadcast", n_nodes_v, self.scratch["n_nodes"]))

        zero_const_v = self.alloc_scratch("zero_const_v", VLEN)
        self.add("valu", ("vbroadcast", zero_const_v, self.scratch_const(0)))
        one_const_v = self.alloc_scratch("one_const_v", VLEN)
        self.add("valu", ("vbroadcast", one_const_v, self.scratch_const(1)))
        two_const_v = self.alloc_scratch("two_const_v", VLEN)
        self.add("valu", ("vbroadcast", two_const_v, self.scratch_const(2)))

        # Broadcast hash constants
        for engine, slot in self.alloc_hash_values():
            self.add(engine, slot)

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))

        tmp1 = self.alloc_scratch("tmp1", VLEN)
        tmp2 = self.alloc_scratch("tmp2", VLEN)
        tmp_idx = self.alloc_scratch("tmp_idx", VLEN)
        tmp_val = self.alloc_scratch("tmp_val", VLEN)
        tmp_node_val = self.alloc_scratch("tmp_node_val", VLEN)
        tmp_idx_addr = self.alloc_scratch("tmp_idx_addr", VLEN)
        tmp_val_addr = self.alloc_scratch("tmp_val_addr", VLEN)
        tmp_node_addr = self.alloc_scratch("tmp_node_addr", VLEN)

        body = []  # array of slots

        for i in range(0, batch_size, VLEN):
            i_const = self.scratch_const(i)
            # idx = mem[inp_indices_p + i]
            body.append(("valu", ("+", tmp_idx_addr, inp_indices_p_v, i_const)))
            body.append(("load", ("vload", tmp_idx, tmp_idx_addr)))
            # val = mem[inp_values_p + i]
            body.append(("valu", ("+", tmp_val_addr, inp_values_p_v, i_const)))
            body.append(("load", ("vload", tmp_val, tmp_val_addr)))

            for round in range(rounds):
                body.append(("debug", ("vcompare", tmp_idx, [(round, i+j, "idx") for j in range(VLEN)])))
                body.append(("debug", ("vcompare", tmp_val, [(round, i+j, "val") for j in range(VLEN)])))
                # node_val = mem[forest_values_p + idx]
                body.append(("valu", ("+", tmp_node_addr, forest_values_p_v, tmp_idx)))
                for j in range(VLEN):
                    body.append(("load", ("load_offset", tmp_node_val, tmp_node_addr, j)))
                body.append(("debug", ("vcompare", tmp_node_val, [(round, i+j, "node_val") for j in range(VLEN)])))
                # val = myhash(val ^ node_val)
                body.append(("valu", ("^", tmp_val, tmp_val, tmp_node_val)))
                body.extend(self.build_hash(tmp_val, tmp1, tmp2, round, i))
                body.append(("debug", ("vcompare", tmp_val, [(round, i+j, "hashed_val") for j in range(VLEN)])))
                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                body.append(("valu", ("%", tmp1, tmp_val, two_const_v)))
                body.append(("valu", ("multiply_add", tmp_idx, tmp_idx, two_const_v, one_const_v)))
                body.append(("valu", ("+", tmp_idx, tmp_idx, tmp1)))
                body.append(("debug", ("vcompare", tmp_idx, [(round, i+j, "next_idx") for j in range(VLEN)])))
                # idx = 0 if idx >= n_nodes else idx
                body.append(("valu", ("<", tmp1, tmp_idx, n_nodes_v)))
                body.append(("valu", ("*", tmp_idx, tmp_idx, tmp1)))
                body.append(("debug", ("vcompare", tmp_idx, [(round, i+j, "wrapped_idx") for j in range(VLEN)])))

            # mem[inp_indices_p + i] = idx
            body.append(("valu", ("+", tmp_idx_addr, inp_indices_p_v, i_const)))
            body.append(("store", ("vstore", tmp_idx_addr, tmp_idx)))
            # mem[inp_values_p + i] = val
            body.append(("valu", ("+", tmp_val_addr, inp_values_p_v, i_const)))
            body.append(("store", ("vstore", tmp_val_addr, tmp_val)))

        body_instrs = self.build(body)
        self.instrs.extend(body_instrs)
        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
