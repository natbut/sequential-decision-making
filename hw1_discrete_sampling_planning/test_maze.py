import numpy as np
from maze import Maze2D, Maze4D

if __name__ == "__main__":
    m = Maze2D.from_pgm('maze1.pgm')
    print('Goal: {}'.format(m.goal_state))
    print('Goal index: {}'.format(m.goal_index))
    print('Test state_from_index: {}'.format(m.state_from_index(m.get_goal())))
    print('Test index_from_state: {}'.format(m.index_from_state(m.goal_state)))
    print('Neighbors of start position:')
    print([m.state_from_index(pos) for pos in m.get_neighbors(m.start_index)])
    path = np.array([(0,0),(0,1),(0,2),(1,2),(2,2)])
    for x in range(m.cols):
        for y in range(m.rows):
            state = (x,y)
            assert m.state_from_index(m.index_from_state(state))==state, "Mapping incorrect for state: {state}".format(state=state)
    m.plot_path(path, 'Maze2D')

    m2 = Maze4D.from_pgm('maze2.pgm')
    for x in range(m2.cols):
        for y in range(m2.rows):
            for dx in range(3):
                for dy in range(3):
                    state = (x,y,dx,dy)
                    assert m2.state_from_index(m2.index_from_state(state))==state, "Mapping incorrect for state: {state}".format(state=state)
    print('Goal: {}'.format(m2.goal_state))
    print('Goal index: {}'.format(m2.goal_index))
    print('Test state_from_index: {}'.format(m.state_from_index(m2.get_goal())))
    print('Test index_from_state: {}'.format(m.index_from_state(m2.goal_state)))
    print('Neighbors of start position:')
    print([m2.state_from_index(pos) for pos in m2.get_neighbors(m2.start_index)])
    path = np.array([(0,0),(0,1),(0,2),(1,2),(2,2)])

    # m2.plot_path(path, 'Maze4D')

