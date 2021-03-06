import numpy as np
from ortools.sat.python import cp_model
import logging

logger = logging.getLogger('__main__')

def solve_puzzle(start_numbers: dict, inequals:dict, puzzle_size:int = 5) -> int:
    """Solve a puzzle

    :param start_numbers: locations of any numbers in the puzzle
    :param inequals: locations of any inequalities in the puzzle
    :param puzzle_size: size of the puzzle, defaults to 5
    :return: final status of ortools solver
    """
    # define model
    model = cp_model.CpModel()

    # define puzzle grid with starting numbers in 1D by row
    grid = np.array([model.NewIntVar(start_numbers.get(i, 1), 
                                     start_numbers.get(i, 5), 
                                     f'{i}') for i in range(puzzle_size ** 2)])
    # make 2D
    grid = grid.reshape((puzzle_size, -1))
    
    # add constraint to all rows and column -- all different
    for row in grid:
        model.AddAllDifferent(row)
    
    for column in grid.T:
        model.AddAllDifferent(column)
    
    # add inequality constraints
    for i in inequals:
        model.Add(grid[i[0]] < grid[i[1]])
    
    # define solver
    solver = cp_model.CpSolver()
    solution_printer = DiagramPrinter(grid, inequals)
    
    status = solver.SearchForAllSolutions(model, solution_printer)
    logger.debug(f'solver status = {status}')
    
    print(f'Solutions found : {solution_printer.solution_count()}')
    
    return status


class DiagramPrinter(cp_model.CpSolverSolutionCallback):
    """Print solutions as a callback"""
    def __init__(self, variables, ineqs):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solution_count = 0
        self.__ineqs = ineqs
    
    def OnSolutionCallback(self):
        self.__solution_count += 1
        
        # define inequality symbols
        tb = 'ʌ'  # u'\u22C0' # ⋀ v
        bt = 'v'  # u'\u22C1' # ⋁ ʌ
        lr = '<'
        rl = '>'
        
        # create copy to alter
        ineqs = self.__ineqs.copy()
        
        print('+---+---+---+---+---+')
        
        for i, row in enumerate(self.__variables):
            # if i == 1: break
            scope_ineqs = [ineq for ineq in ineqs if ineq[0][0] == i or ineq[1][0] == i]
            
            row = row.tolist()
            vals = [self.Value(val) for val in row]
            
            # row / col subs are indisces to subsitute for inequalitiess
            col_sep = ['|', ' '] + list(' | '.join(str(x) for x in vals)) + [' ', '|']
            col_subs = [4, 8, 12, 16]
            
            row_sep = list('+---+---+---+---+---+')
            row_subs = [2, 6, 10, 14, 18]
            
            # print out inequalities
            for ineq in scope_ineqs:
                lower = ineq[0]
                higher = ineq[1]
                
                if lower[0] == higher[0]:  # if in the same row
                    if lower[1] > higher[1]:  # if lower end is to the right
                        col_sep[col_subs[min(lower[1], higher[1])]] = rl
                    elif lower[1] < higher[1]:  # if lower end is to the left
                        col_sep[col_subs[min(lower[1], higher[1])]] = lr
                    
                elif i != 4 and lower[1] == higher[1]:  # if in the same column, don't need for final row
                    if lower[0] > higher[0]:  # if the lower end is on the bottom column
                        row_sep[row_subs[lower[1]]] = bt
                    elif lower[0] < higher[0]:  # if the lower end is on top column
                        row_sep[row_subs[lower[1]]] = tb
                
                # remove entry at end of loop to avoid double printing
                ineqs.remove(ineq)
                       
            print(''.join(col_sep))
            print(''.join(row_sep))

    def solution_count(self):
        return self.__solution_count
