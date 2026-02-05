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

    # TODO: Allow buffer to exceed slot limits and automatically pack them when flushing.
    def add_to_buffer(self, engine, slot):
        if engine not in self.buffer:
            self.buffer[engine] = []
        if len(self.buffer[engine]) < SLOT_LIMITS[engine]:
            self.buffer[engine].append(slot)
            return
        raise Exception(f"{engine} is full") 
    
    def flush_buffer(self):
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

    def scratch_const(self, val, name=None, buffered=False):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            if buffered:
                self.add_to_buffer("load", ("const", addr, val))
            else:
                self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def add_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            self.add_to_buffer("valu", (op1, tmp1, val_hash_addr, self.scratch[f"{val1}_vec"]))
            self.add_to_buffer("valu", (op3, tmp2, val_hash_addr, self.scratch[f"{val3}_vec"]))
            self.flush_buffer()
            self.add("valu", (op2, val_hash_addr, tmp1, tmp2))
            self.add("debug", ("vcompare", val_hash_addr, [(round, j, "hash_stage", hi) for j in range(i, i+VLEN)]))
    
    # Allocate vectors of hash-related constants to utilize vector instructions
    def init_hash_values(self):
        for (op1, val1, op2, op3, val3) in HASH_STAGES:
            val1_const = self.scratch_const(val1, None, True)
            val3_const = self.scratch_const(val3, None, True)
            self.flush_buffer()
            val1_vec = self.alloc_scratch(f"{val1}_vec", VLEN)
            val3_vec = self.alloc_scratch(f"{val3}_vec", VLEN)
            self.add_to_buffer("valu", ("vbroadcast", val1_vec, val1_const))
            self.add_to_buffer("valu", ("vbroadcast", val3_vec, val3_const))
            self.flush_buffer()

    """
    - Baseline (147734)
    - Split `tmp_addr` to remove unnecessary repetitions. (139542)
    - Use `vload` for initial setup. (139529)
    - Remove unneccessary comparison during index update. (135433)
    - Aggregate independent instrs to buffer (119049)
    - Aggregate hash instrs (94473)
    - Aggregate stores with next loads (90378)
    - Apply vector instructions (16658)
    - Switch loop order and reuse round-independent values (12338)
    - Aggregate scratch_const operations (12210)
    - Loop unrolling x 4 (7073)
    - Apply vload and trivial pipelines (6457)
    - Pack load_offsets with hashing valus (5689)
    - Use multiply_add for next_idx calculation (5305)
    """
    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Scalar implementation using only scalar ALU and load/store.
        """
        LOOP_UNROLL = 4 

        tmp1 = [self.alloc_scratch(f"tmp1_{i}", VLEN) for i in range(LOOP_UNROLL)]
        tmp2 = [self.alloc_scratch(f"tmp2_{i}", VLEN) for i in range(LOOP_UNROLL)]
        tmp3 = [self.alloc_scratch(f"tmp3_{i}", VLEN) for i in range(LOOP_UNROLL)]
    
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
        for i in range(0, batch_size, SLOT_LIMITS["load"]):
            for j in range(SLOT_LIMITS["load"]):
                self.scratch_const(i+j, None, True)
            self.flush_buffer()
        
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
        tmp_idx = [self.alloc_scratch(f"tmp_idx_{i}", VLEN) for i in range(LOOP_UNROLL)]
        tmp_val = [self.alloc_scratch(f"tmp_val_{i}", VLEN) for i in range(LOOP_UNROLL)]
        tmp_node_val = [self.alloc_scratch(f"tmp_node_val_{i}", VLEN) for i in range(LOOP_UNROLL)]

        tmp_iaddr = [self.alloc_scratch(f"tmp_iaddr_{i}", VLEN) for i in range(LOOP_UNROLL)]
        tmp_vaddr = [self.alloc_scratch(f"tmp_vaddr_{i}", VLEN) for i in range(LOOP_UNROLL)]
        tmp_naddr = [self.alloc_scratch(f"tmp_naddr_{i}", VLEN) for i in range(LOOP_UNROLL)]

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

        self.flush_buffer()

        self.add("valu", ("vbroadcast", n_nodes_vec, self.scratch["n_nodes"]))

        for i in range(0, batch_size, VLEN*LOOP_UNROLL):
            i_const = [self.scratch_const(i+VLEN*t) for t in range(LOOP_UNROLL)]

            # idx = mem[inp_indices_p + i], val = mem[inp_values_p + i]
            for t in range(LOOP_UNROLL):
                self.add_to_buffer("valu", ("+", tmp_iaddr[t], indices_vec, i_const[t]))
                self.add_to_buffer("valu", ("+", tmp_vaddr[t], values_vec, i_const[t]))
                if t > 0:
                    self.add_to_buffer("load", ("vload", tmp_idx[t-1], tmp_iaddr[t-1]))
                    self.add_to_buffer("load", ("vload", tmp_val[t-1], tmp_vaddr[t-1]))
                self.flush_buffer()

            self.add_to_buffer("load", ("vload", tmp_idx[-1], tmp_iaddr[-1]))
            self.add_to_buffer("load", ("vload", tmp_val[-1], tmp_vaddr[-1]))
            self.flush_buffer()

            for round in range(rounds):
                for t in range(LOOP_UNROLL):
                    self.add("debug", ("vcompare", tmp_idx[t], [(round, j, "idx") for j in range(i+VLEN*t, i+VLEN*t+VLEN)]))
                    self.add("debug", ("vcompare", tmp_val[t], [(round, j, "val") for j in range(i+VLEN*t, i+VLEN*t+VLEN)]))

                # node_val = mem[forest_values_p + idx]
                for t in range(LOOP_UNROLL):
                    self.add_to_buffer("valu", ("+", tmp_naddr[t], forest_values_vec, tmp_idx[t]))
                self.flush_buffer()

                for t in range(LOOP_UNROLL):
                    if t == 0:
                        for j in range(0, VLEN, SLOT_LIMITS["load"]):
                            for k in range(SLOT_LIMITS["load"]):
                                self.add_to_buffer("load", ("load_offset", tmp_node_val[t], tmp_naddr[t], j+k))
                            self.flush_buffer()

                    self.add("debug", ("vcompare", tmp_node_val[t], [(round, j, "node_val") for j in range(i+VLEN*t, i+VLEN*t+VLEN)]))

                    # val = myhash(val ^ node_val)
                    if t == 0:
                        self.add_to_buffer("valu", ("^", tmp_val[t], tmp_val[t], tmp_node_val[t]))
                        self.flush_buffer()

                    # self.add_hash(tmp_val[t], tmp1[t], tmp2[t], round, i + t*VLEN)
                    # t = 0: Hash for t = 0 and load tmp_val[1], [2], [3]
                    # t = 1: Calculate all leftover cases
                    # t = 2, 3: Pass since there are no load or hashing left
                    for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                        if t == 0:
                        # tmp_node_val[1] is ready
                            if hi == 2:
                                self.add_to_buffer("valu", ("^", tmp_val[1], tmp_val[1], tmp_node_val[1])) 
                            # tmp_node_val[2] is ready
                            if hi == 4:
                                self.add_to_buffer("valu", ("^", tmp_val[2], tmp_val[2], tmp_node_val[2]))
                            self.add_to_buffer("valu", (op1, tmp1[t], tmp_val[t], self.scratch[f"{val1}_vec"]))
                            self.add_to_buffer("valu", (op3, tmp2[t], tmp_val[t], self.scratch[f"{val3}_vec"]))
                            if hi < 2:
                                self.add_to_buffer("load", ("load_offset", tmp_node_val[1], tmp_naddr[1], 4*hi))
                                self.add_to_buffer("load", ("load_offset", tmp_node_val[1], tmp_naddr[1], 4*hi+1))
                            if hi >= 2 and hi < 4:
                                self.add_to_buffer("load", ("load_offset", tmp_node_val[2], tmp_naddr[2], 4*(hi-2)))
                                self.add_to_buffer("load", ("load_offset", tmp_node_val[2], tmp_naddr[2], 4*(hi-2)+1))
                            if hi >= 4:
                                self.add_to_buffer("load", ("load_offset", tmp_node_val[3], tmp_naddr[3], 4*(hi-4)))
                                self.add_to_buffer("load", ("load_offset", tmp_node_val[3], tmp_naddr[3], 4*(hi-4)+1))
                            self.flush_buffer()
                            self.add_to_buffer("valu", (op2, tmp_val[t], tmp1[t], tmp2[t]))
                            if hi < 2:
                                self.add_to_buffer("load", ("load_offset", tmp_node_val[1], tmp_naddr[1], 4*hi+2))
                                self.add_to_buffer("load", ("load_offset", tmp_node_val[1], tmp_naddr[1], 4*hi+3))
                            if hi >= 2 and hi < 4:
                                self.add_to_buffer("load", ("load_offset", tmp_node_val[2], tmp_naddr[2], 4*(hi-2)+2))
                                self.add_to_buffer("load", ("load_offset", tmp_node_val[2], tmp_naddr[2], 4*(hi-2)+3))
                            if hi >= 4:
                                self.add_to_buffer("load", ("load_offset", tmp_node_val[3], tmp_naddr[3], 4*(hi-4)+2))
                                self.add_to_buffer("load", ("load_offset", tmp_node_val[3], tmp_naddr[3], 4*(hi-4)+3))
                            self.flush_buffer()
                        
                        # Note that all tmp_node_vals are ready
                        elif t == 1:
                            for k in range(1, 4):
                                self.add_to_buffer("valu", (op1, tmp1[k], tmp_val[k], self.scratch[f"{val1}_vec"]))
                                self.add_to_buffer("valu", (op3, tmp2[k], tmp_val[k], self.scratch[f"{val3}_vec"]))
                            self.flush_buffer()
                            for k in range(1, 4):
                                self.add_to_buffer("valu", (op2, tmp_val[k], tmp1[k], tmp2[k]))
                            self.flush_buffer()
                            for k in range(1, 4): 
                                self.add("debug", ("vcompare", tmp_val[k], [(round, j, "hash_stage", hi) for j in range(i+VLEN*k, i+VLEN*k+VLEN)]))
                        
                        else:
                            pass
                    
                    if t == 0:
                        # tmp_node_val[3] is ready
                        self.add_to_buffer("valu", ("^", tmp_val[3], tmp_val[3], tmp_node_val[3]))
                        self.flush_buffer()

                    self.add("debug", ("vcompare", tmp_val[t], [(round, j, "hashed_val") for j in range(i+VLEN*t, i+VLEN*t+VLEN)]))

                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                for t in range(LOOP_UNROLL):
                    self.add_to_buffer("valu", ("%", tmp1[t], tmp_val[t], two_vec))
                self.flush_buffer()

                for t in range(LOOP_UNROLL):
                    self.add_to_buffer("valu", ("+", tmp1[t], tmp1[t], one_vec))
                self.flush_buffer()

                for t in range(LOOP_UNROLL):
                    self.add_to_buffer("valu", ("multiply_add", tmp_idx[t], tmp_idx[t], two_vec, tmp1[t]))
                self.flush_buffer()

                for t in range(LOOP_UNROLL):
                    self.add("debug", ("vcompare", tmp_idx[t], [(round, j, "next_idx") for j in range(i+VLEN*t, i+VLEN*t+VLEN)]))

                # idx = 0 if idx >= n_nodes else idx
                for t in range(LOOP_UNROLL):
                    self.add_to_buffer("valu", ("<", tmp1[t], tmp_idx[t], n_nodes_vec))
                self.flush_buffer()

                for t in range(LOOP_UNROLL):
                    self.add_to_buffer("flow", ("vselect", tmp_idx[t], tmp1[t], tmp_idx[t], zero_vec))
                    # mem[inp_indices_p + i] = idx
                    # mem[inp_values_p + i] = val
                    if t > 0:
                        self.add_to_buffer("store", ("vstore", tmp_iaddr[t-1], tmp_idx[t-1]))
                        self.add_to_buffer("store", ("vstore", tmp_vaddr[t-1], tmp_val[t-1]))
                    self.flush_buffer()
                    self.add("debug", ("vcompare", tmp_idx[t], [(round, j, "wrapped_idx") for j in range(i+VLEN*t, i+VLEN*t+VLEN)]))

                self.add_to_buffer("store", ("vstore", tmp_iaddr[-1], tmp_idx[-1]))
                self.add_to_buffer("store", ("vstore", tmp_vaddr[-1], tmp_val[-1]))
                self.flush_buffer() 

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
