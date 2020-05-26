# cell_automata
Some cellular automata I was playing around with during my B21 fellowship.

A description of each file:

  rule_feedback:

  What happens when the rule applied to each step of a cellular automaton depends on the state of the automaton?
  This is possible because both rule and state are represented by sets of integers (generalized beyond elementary
  cellular automata from Wolfram's "Cellular Automata as Simple Self-Organizing Systems").

  Shows a plot with this feedback on the left, and a constant rule (equal to the first state) on the right.

  Parameters such as number of time steps (n), number of possible cell values (number_of_types), and number of
  cells grouped together when the rule is applied (group_size) analogous to range of the rule.
  Note, the size of the state in the simulation (the number of cells) is restricted to number_of_types ^ group_size.

  The boundaries are periodic (think Pacman).

