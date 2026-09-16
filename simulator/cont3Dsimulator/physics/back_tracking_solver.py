import numpy as np


class State():
    def __init__(self, **kwargs):
        pass

class BackTrackSolver():
    def __init__(self,**kwargs):
        self.state_list = [self.root(**kwargs)]
        self.solved = False
    def reject(self, state:State):
        # Implement logic to determine if the current state should be rejected
        return False
    def accept(self, state:State):
        # Implement logic to determine if the current state is a solution
        return False
    def root(self, **kwargs):
        # Implement logic to create the initial state
        return State(**kwargs)
    def next(self, state:State):
        # Implement logic to generate the next states from the current state
        return []
    
    def backtrack(self, current_state):
        if self.reject(current_state):
            return None
        if self.accept(current_state):
            self.solved = True
            return current_state
        self.state_list.append(self.next(current_state))
        self.backtrack(self.state_list[-1])
        if self.solved:
            return self.state_list[-1]
        self.state_list.pop()
        return None

class PhysicsState(State):
    def __init__(self, level, blocks, contacts, **kwargs):
        super().__init__(**kwargs)
        self.level = level
        self.blocks = blocks
        self.contacts = contacts
    def is_falling(self):
        # Implement logic to determine if any block is falling
        return False
class PhysicsBackTrackSolver(BackTrackSolver):
    def __init__(self,blocks:np.ndarray, contacts:np.ndarray, **kwargs):
        self.blocks = blocks
        self.contacts = contacts
        super().__init__(**kwargs)
    def reject(self, state:PhysicsState):
        for block in state.blocks:
            if block.is_falling():
                return True
        return False
    def accept(self, state:PhysicsState):
        # Reached the ground level
        if state.level == 0:
            return True
        return False
    def root(self, **kwargs):
        # Implement logic to create the initial physics state
        return PhysicsState(**kwargs)
    def next(self, state:PhysicsState):
        # Implement logic to generate the next physics states from the current state
        return []

    
if __name__ == "__main__":
    import masongraph_envs
    from simulator.cont3Dsimulator.batchsim import activeUI
    env = masongraph_envs.StraightMasonGraphCover(n_batch=1,max_blocks=80,use_batchsolver=True,physics_on=True,tol=1e-2,ground_gen='around',autoreset=False)
    env.reset()
    activeUI(env.sim)
    phymodel = PhysicsBackTrackSolver(blocks=env.sim.assembly_sequence, contacts=env.sim.contacts)
    direction = np.zeros((env.n_batch,3))
    direction[:,2]=-1

    action = {"robot_id":np.full(env.n_batch,0,dtype=int),
              "block_type":np.zeros(env.n_batch,dtype=int),
              "support_block": np.full(env.n_batch,21,dtype=int),
              "center_offset":np.array([[0,0,0]]),
              'rotation_offset':np.tile(np.array([[0,0]]),(env.n_batch,1)),
              "direction":direction,
            }
    obs, reward, terminated, truncated,info = env.step(action)
