def my_render_frozen_lake(self):
    
    original = True # original frozen lake letter
    
    if self.lastaction is not None:
        print(f"  ({['Left', 'Down', 'Right', 'Up'][self.lastaction]})")
    else:
        print("")
    
    desc = self.desc.tolist()

    row, col = self.s // self.ncol, self.s % self.ncol
    desc = [[c.decode("utf-8") for c in line] for line in desc]

    for i, line in enumerate(desc):
        for j, c in enumerate(line):
            if (i,j) == (row, col):
                # print("P", end="")
                print("🧑", end="")
            elif desc[i][j] == "G":
                # print("G", end="")
                print("⛳️", end="")
            elif desc[i][j] == "H":
                # print("H" if original else "O", end="")
                print("🕳", end="")
            else:
                # print("F" if original else ".", end="")
                print("🧊", end="")
        print()

def fix_frozen_lake_render(env):
    env.render = type(env.render)(my_render_frozen_lake, env)
    