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
        # Temporary buffer that aggregates slots per engine
        self.buffer = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add_to_buffer(self, engine, slot):
        if engine not in self.buffer:
            self.buffer[engine] = []
        if len(self.buffer[engine]) < SLOT_LIMITS[engine]:
            self.buffer[engine].append(slot)
            return
        raise Exception(f"{engine} is full") 
    
    def buffer_to_instrs(self):
        self.instrs.append(self.buffer)
        self.buffer = {}

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

    def add_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            self.add_to_buffer("valu", (op1, tmp1, val_hash_addr, self.scratch[f"{val1}_vec"]))
            self.add_to_buffer("valu", (op3, tmp2, val_hash_addr, self.scratch[f"{val3}_vec"]))
            self.buffer_to_instrs()
            self.add("valu", (op2, val_hash_addr, tmp1, tmp2))
            self.add("debug", ("vcompare", val_hash_addr, [(round, j, "hash_stage", hi) for j in range(i, i+VLEN)]))
    
    # Allocate vectors of hash-related constants to utilize vector instructions
    def init_hash_values(self):
        for (op1, val1, op2, op3, val3) in HASH_STAGES:
            val1_const = self.scratch_const(val1)
            val3_const = self.scratch_const(val3)
            val1_vec = self.alloc_scratch(f"{val1}_vec", VLEN)
            val3_vec = self.alloc_scratch(f"{val3}_vec", VLEN)
            self.add_to_buffer("valu", ("vbroadcast", val1_vec, val1_const))
            self.add_to_buffer("valu", ("vbroadcast", val3_vec, val3_const))
            self.buffer_to_instrs()

    """
    - Baseline (147734)
    - Split `tmp_addr` to remove unnecessary repetitions. (139542)
    - Use `vload` for initial setup. (139529)
    - Remove unneccessary comparison during index update. (135433)
    - Aggregate independent instrs to buffer (119049)
    - Aggregate hash instrs (94473)
    - Aggregate stores with next loads (90378)
    - Apply vector instructions (16658)
    """
    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Scalar implementation using only scalar ALU and load/store.
        """
        tmp1 = self.alloc_scratch("tmp1", VLEN)
        tmp2 = self.alloc_scratch("tmp2", VLEN)
        tmp3 = self.alloc_scratch("tmp3", VLEN)
    
        # Scratch space addresses
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
            "placeholder"
        ]
        for v in init_vars:
            self.alloc_scratch(v)

        self.add("load", ("vload", self.scratch["rounds"], 0))
        # for i, v in enumerate(init_vars):
        #     self.add("load", ("const", tmp1, i))
        #     self.add("load", ("load", self.scratch[v], tmp1))
        
        # Load all const at once to make them adjacent in scratch space
        for i in range(batch_size):
            self.scratch_const(i)
        
        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))

        # Scalar scratch registers
        tmp_idx = self.alloc_scratch("tmp_idx", VLEN)
        tmp_val = self.alloc_scratch("tmp_val", VLEN)
        tmp_node_val = self.alloc_scratch("tmp_node_val", VLEN)

        tmp_iaddr = self.alloc_scratch("tmp_iaddr", VLEN)
        tmp_vaddr = self.alloc_scratch("tmp_vaddr", VLEN)
        tmp_naddr = self.alloc_scratch("tmp_naddr", VLEN)

        indices_vec = self.alloc_scratch("indices_vec", VLEN)
        values_vec = self.alloc_scratch("values_vec", VLEN)
        forest_values_vec = self.alloc_scratch("forest_values_vec", VLEN)
        n_nodes_vec = self.alloc_scratch("n_nodes_vec", VLEN)

        zero_vec = self.alloc_scratch("zero_vec", VLEN)
        one_vec = self.alloc_scratch("one_vec", VLEN)
        two_vec = self.alloc_scratch("two_vec", VLEN)
       
        # Broadcast preparation for vector instructions 
        self.init_hash_values()
        
        self.add_to_buffer("valu", ("vbroadcast", zero_vec, zero_const))
        self.add_to_buffer("valu", ("vbroadcast", one_vec, one_const))
        self.add_to_buffer("valu", ("vbroadcast", two_vec, two_const))
        self.add_to_buffer("valu", ("vbroadcast", indices_vec, self.scratch["inp_indices_p"]))
        self.add_to_buffer("valu", ("vbroadcast", values_vec, self.scratch["inp_values_p"] ))
        self.add_to_buffer("valu", ("vbroadcast", forest_values_vec, self.scratch["forest_values_p"]))

        self.buffer_to_instrs()

        self.add("valu", ("vbroadcast", n_nodes_vec, self.scratch["n_nodes"]))

        for round in range(rounds):
            for i in range(0, batch_size, VLEN):
                i_const = self.scratch_const(i)

                # idx = mem[inp_indices_p + i], val = mem[inp_values_p + i]
                self.add_to_buffer("valu", ("+", tmp_iaddr, indices_vec, i_const))
                self.add_to_buffer("valu", ("+", tmp_vaddr, values_vec, i_const))

                self.buffer_to_instrs()

                for j in range(VLEN):
                    self.add_to_buffer("load", ("load_offset", tmp_idx, tmp_iaddr, j))
                    self.add_to_buffer("load", ("load_offset", tmp_val, tmp_vaddr, j))

                    self.buffer_to_instrs()
                
                self.add("debug", ("vcompare", tmp_idx, [(round, j, "idx") for j in range(i, i+VLEN)]))
                self.add("debug", ("vcompare", tmp_val, [(round, j, "val") for j in range(i, i+VLEN)]))

                # node_val = mem[forest_values_p + idx]
                self.add("valu", ("+", tmp_naddr, forest_values_vec, tmp_idx))

                for j in range(0, VLEN, SLOT_LIMITS["load"]):
                    for k in range(SLOT_LIMITS["load"]):
                        self.add_to_buffer("load", ("load_offset", tmp_node_val, tmp_naddr, j + k))
                    self.buffer_to_instrs()

                self.add("debug", ("vcompare", tmp_node_val, [(round, j, "node_val") for j in range(i, i+VLEN)]))

                # val = myhash(val ^ node_val)
                self.add("valu", ("^", tmp_val, tmp_val, tmp_node_val))
                self.add_hash(tmp_val, tmp1, tmp2, round, i)
                self.add("debug", ("vcompare", tmp_val, [(round, j, "hashed_val") for j in range(i, i+VLEN)]))

                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                self.add_to_buffer("valu", ("%", tmp1, tmp_val, two_vec))
                self.add_to_buffer("valu", ("*", tmp_idx, tmp_idx, two_vec))
                self.buffer_to_instrs()
                self.add("flow", ("vselect", tmp3, tmp1, two_vec, one_vec))
                self.add("valu", ("+", tmp_idx, tmp_idx, tmp3))
                self.add("debug", ("vcompare", tmp_idx, [(round, j, "next_idx") for j in range(i, i+VLEN)]))

                # idx = 0 if idx >= n_nodes else idx
                self.add("valu", ("<", tmp1, tmp_idx, n_nodes_vec))
                self.add("flow", ("vselect", tmp_idx, tmp1, tmp_idx, zero_vec))
                self.add("debug", ("vcompare", tmp_idx, [(round, j, "wrapped_idx") for j in range(i, i+VLEN)]))

                # mem[inp_indices_p + i] = idx
                self.add_to_buffer("store", ("vstore", tmp_iaddr, tmp_idx))

                # mem[inp_values_p + i] = val
                self.add_to_buffer("store", ("vstore", tmp_vaddr, tmp_val))

                # self.buffer_to_instrs()

        self.buffer_to_instrs()
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
