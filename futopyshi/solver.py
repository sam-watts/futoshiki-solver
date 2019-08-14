import numpy as np
from ortools.sat.python import cp_model
import time

def create_model(puzzle_size, start_numbers, ineqs=None):
    # define model
    model = cp_model.CpModel()

    # define puzzle grid with starting numbers
    grid = np.array([model.NewIntVar(start_numbers.get(i, 1), 
                                     start_numbers.get(i, 5), 
                                     f'{i}') for i in range(puzzle_size ** 2)])
    # make 2D
    grid = grid.reshape((puzzle_size, -1))
    
    # need to add constraint to all rows and column -- all different
    for row in grid:
        model.AddAllDifferent(row)
    
    for column in grid.T:
        model.AddAllDifferent(column)
        
    # Create inequality constraints  
    # data format as tuples, 
    # * `[0]` = is on the lower side of the inequality
    # * `[1]` = is on the higher side of the inequality
    
    # as 1D coordinates, deprecated
    # ineqs = [(1,0), (3,2), (4,3), (18,19), (20,21), (21,22)]
    
    # as 2D coordinates, currently in use
    ineqs = [
        ((0,1), (0,0)),
        ((0,3), (0,2)),
        ((0,4), (0,3)),
        ((3,3), (3,4)),
        ((4,0), (4,1)),
        ((4,1), (4,2))
        ]
    
    # add inequality constraints
    for i in ineqs:
        model.Add(grid[i[0]] < grid[i[1]])
    
    # define solver
    solver = cp_model.CpSolver()
    
    return solver

class DiagramPrinter(cp_model.CpSolverSolutionCallback):
    def __init__(self, variables, ineqs):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solution_count = 0
        self.__ineqs = ineqs
    
    def OnSolutionCallback(self):
        self.__solution_count += 1
 
        for row in self.__variables:
            row = row.tolist()
            vals = [self.Value(val) for val in row]
            print("+---+---+---+---+---+")
            print('| ', end = '')
            print(' | '.join(str(x) for x in vals), end = '')
            print(' |')
        
        print("+---+---+---+---+---+")
    
    def SolutionCount(self):
        return self.__solution_count

def solve_model(solver):
    solution_printer = DiagramPrinter(grid, ineqs)
    start = time.time()
    status = solver.SearchForAllSolutions(model, solution_printer)
    end = time.time()
    print(f'Time elapsed: {round(end - start, 4)}')
    print('Solutions found : %i' % solution_printer.SolutionCount())