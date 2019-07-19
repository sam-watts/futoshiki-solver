    '||''''|            ||                 '||          '||              .|'''|         '||`                      
     ||  .              ||                  ||      ''   ||       ''     ||              ||                       
     ||''|   '||  ||` ''||''  .|''|, (''''  ||''|,  ||   || //`   ||     `|'''|, .|''|,  ||  \\  // .|''|, '||''| 
     ||       ||  ||    ||    ||  ||  `'')  ||  ||  ||   ||<<     ||      .   || ||  ||  ||   \\//  ||..||  ||    
    .||.      `|..'|.   `|..' `|..|' `...' .||  || .||. .|| \\.  .||.     |...|' `|..|' .||.   \/   `|...  .||.   
                                                                                                              
                                                                                                       
An [OR-TOOLS](https://developers.google.com/optimization/) based solver of Futoshiki (不等式) puzzles


### Initial Notes

pseudo code
use 2d numpy array for position of integers

how to input position of inequalities?
some kind of recognition from an image would be sick! 
this could combine hand written / printed digit recognition - eg. if you have half finished a puzzle
but this would need to only focus on numerals in boxes + inequality symbols, ie. notes must be ignored

could just be easier to use some kind of input from command line using basic coordinate system? 
