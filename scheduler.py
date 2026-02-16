from collections import defaultdict
from problem import SLOT_LIMITS, VLEN

class InstructionScheduler:
    def __init__(self):
        self.reg_ready_time = defaultdict(int)
        self.schedule_map = defaultdict(lambda: defaultdict(list))

    def _expand_range(self, start_addr, length=1):
        return list(range(start_addr, start_addr + length))

    def _parse_dependencies(self, engine, slot):
        reads = []
        writes = []
        
        def R(val, length=1): reads.extend(self._expand_range(val, length))
        def W(val, length=1): writes.extend(self._expand_range(val, length))
        
        op = slot[0]

        if engine == "alu":
            _, dest, src1, src2 = slot
            W(dest)
            R(src1); R(src2)

        elif engine == "valu":
            if op == "vbroadcast":
                _, dest, src = slot
                W(dest, VLEN)
                R(src, 1) 
            elif op == "multiply_add":
                _, dest, a, b, c = slot
                W(dest, VLEN)
                R(a, VLEN); R(b, VLEN); R(c, VLEN)
            else:
                # (op, dest, src1, src2)
                _, dest, src1, src2 = slot
                W(dest, VLEN)
                R(src1, VLEN); R(src2, VLEN)

        elif engine == "load":
            if op == "load":
                _, dest, addr = slot
                W(dest, 1)
                R(addr, 1)
            elif op == "vload":
                _, dest, addr = slot
                W(dest, VLEN)
                R(addr, 1) 
            elif op == "load_offset":
                _, dest, addr, offset = slot
                W(dest + offset, 1)
                R(addr + offset, 1)
            elif op == "const":
                _, dest, _ = slot
                W(dest, 1)

        elif engine == "store":
            if op == "store":
                _, addr, src = slot
                R(addr, 1); R(src, 1)
            elif op == "vstore":
                _, addr, src = slot
                R(addr, 1); R(src, VLEN)
        
        elif engine == "flow":
             if op == "select":
                 _, dest, cond, a, b = slot
                 W(dest, 1)
                 R(cond, 1); R(a, 1); R(b, 1)
             elif op == "vselect": 
                 _, dest, cond, a, b = slot
                 W(dest, VLEN)
                 R(cond, VLEN); R(a, VLEN); R(b, VLEN)

        elif engine == "debug":
            if op == "compare":
                _, addr, _ = slot
                R(addr, 1)
            elif op == "vcompare":
                _, addr, _ = slot
                R(addr, VLEN)

        return reads, writes

    def add_instruction(self, engine, slot):
        reads, writes = self._parse_dependencies(engine, slot)
        
        # 1. Calculate the minimum safe cycle based on RAW, WAW dependencies.
        min_cycle = 0
        for r_reg in reads:
            min_cycle = max(min_cycle, self.reg_ready_time[r_reg])
        for w_reg in writes:
            min_cycle = max(min_cycle, self.reg_ready_time[w_reg])

        # 2. Check the quickest available slot.
        current_cycle = min_cycle
        while True:
            current_usage = len(self.schedule_map[current_cycle][engine])
            if current_usage < SLOT_LIMITS[engine]:
                break
            current_cycle += 1
            
        self.schedule_map[current_cycle][engine].append(slot)
        
        # 3. Update the ready time based on writes.
        for w_reg in writes:
            self.reg_ready_time[w_reg] = current_cycle + 1

    def build(self):
        instrs = []
        for _, instr in self.schedule_map.items():
            instrs.append(instr)
        return instrs

def schedule_vliw(slots_list):
    scheduler = InstructionScheduler()
    for engine, slot in slots_list:
        scheduler.add_instruction(engine, slot)
    return scheduler.build()
