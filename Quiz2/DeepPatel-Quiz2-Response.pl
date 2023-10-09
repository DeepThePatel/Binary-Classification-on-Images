%% Domain
%% dig(X): X is a digit.
dig(0). dig(1). dig(2). dig(3). dig(4).
dig(5). dig(6). dig(7). dig(8). dig(9).

%% Constraints
%% An auxiliary predicate to ensure uniqueness.
uniq(T, W, O, F, U, R) :-
    \+ T = W,
    \+ T = O,
    \+ T = F,
    \+ T = U,
    \+ T = R,

    \+ W = O,
    \+ W = F,
    \+ W = U,
    \+ W = R,

    \+ F = O,
    \+ F = U,
    \+ F = R,

    \+ O = U,
    \+ O = R,

    \+ U = R.

%% Main
solve_digits(T, W, F, O, U, R) :-
    %% defining digits
        dig(T), dig(W), dig(F), dig(O),
        dig(U), dig(R),
   
    %% defining T,F to be > 0
    T > 0, F > 0,

    %% defining uniqueness
    uniq(T, W, F, O, U, R),

    TWO is T * 100 + W * 10 + O,
    FOUR is F * 1000 + O * 100 + U * 10 + R,

    TWO + TWO =:= FOUR.

%% Print solution visually
print_solution :-
    solve_digits(T, W, F, O, U, R),
    format("~d~d~d\n", [T, W, O]),
    write("+"), nl, 
    format("~d~d~d\n", [T, W, O]),
    write("-------------\n"),
    format("~d~d~d~d\n", [F, O, U, R]).