/* -*-prolog-*- */
/* PABL modification and adaptation of QMPT0 solver by yuki goto, naoyuki nide. 2017.month(4-5)*/

/*
QMPT0 solver for SWI-Prolog

Usage:
	?- qmpt0_refresh.
		Initial the definition of qmpt0_rule/2 and qmpt0_hyp/2.
	?- qmpt0_add_rule(A <= [B1, ..., Bn]).
		Add a QMPT0 clause A <= B1, ..., Bn to the QMPT0 logic program
		expressed by qmpt0_rule/2.
	?- qmpt0_add_fact(A).
		Equivalent to qmpt0_add_rule(A <= []).
	?- qmpt0_solve(A).
		Try to derive A from current QMPT0 logic program expressed by
		qmpt0_rule/2. It may generate alternative solutions.

Predicates internally used:
	qmpt0_rule(A, [B1, ..., Bn])
		Expresses a QMPT0 clause. A fact is expressed as qmpt0_rule(A,
		[]). Asserted by qmpt0_add_rule/1.
	qmpt0_hyp(A, [B1, ..., Bn])
		A QMPT0 clause asserted during the execution of solver
		algorithm. Users do not have to give or modify it manually.

Execution of sample:
	?- qmpt0_test.
	true.
	?- qmpt0_solve(stray(X)).
	X = yuki ;
	X = kuma ;
	false.
*/


/* operator declaration */
:- op(496, fy, -),
   op(497, xfy, <=).

/* dynamic predicate declaration */
:- dynamic qmpt0_rule/2.
:- dynamic qmpt0_hyp/2.

/* interface of our solver engine (meta predicates) */
qmpt0_add_fact(X):-
    asserta(qmpt0_rule(X,[])).

qmpt0_add_rule(X <= Antecedentlist):-
    asserta(qmpt0_rule(X,Antecedentlist)).

qmpt0_refresh :-
    retractall(qmpt0_rule(_,_)),
    retractall(qmpt0_hyp(_,_)).

qmpt0_solve(X):-
    retractall(qmpt0_hyp(_,_)),
    assertz(qmpt0_hyp(X,[X])),
    solve_process(X,[X]).
    
/* main algorithm of solver */
solve_process(Goal, Subgoalseq) :-
    select(X, Subgoalseq, RestSubgoalseq),
    clause(qmpt0_rule(X, Alist), _),
    append(Alist, RestSubgoalseq, Complexlist),
    list_to_set(Complexlist, NewSubgoalseq),  % Convert list to set to remove duplicates
    \+ clause(qmpt0_hyp(Goal, NewSubgoalseq), _),  % Check if this state has been processed
    (add_qmpt0_hyp(Goal, NewSubgoalseq) -> true ; 
    solve_process(Goal, NewSubgoalseq)).



/* sub algorithm for solver; add some clauses as hypotheses to be solved */
/*Add a new hypothesis if it does not already exist*/
add_qmpt0_hyp(_, []) :- !, true.

add_qmpt0_hyp(Goal, Subgoalseq) :-
    list_to_set(Subgoalseq, UniqueSubgoalseq),  % Ensure uniqueness within the subgoal sequence
     % Check for existence
    assertz(qmpt0_hyp(Goal, UniqueSubgoalseq)),  % Add if not present
    fail.  % Force backtracking to explore other possibilities

add_qmpt0_hyp(Goal, Subgoalseq) :-
    select(X, Subgoalseq, Restgoalseq),
    clause(qmpt0_hyp(Goal, Alist), _),
    select(Z, Alist, Rlist),
    (X = -Z; Z = -X),  % Handling negation or difference
    append(Restgoalseq, Rlist, Newlist),
    list_to_set_alpha(Newlist, NewSubgoalseq),
    add_qmpt0_hyp(Goal, NewSubgoalseq).

/* similar to list_to_set/2 of SWI-Prolog, but equate and unify the terms
   which are alpha-equivalent */
list_to_set_alpha(L,M):-
    reverse(L,LR),
    list_to_set_alpha_aux(LR,MR),
    reverse(MR,M).
alpha_member(X,[X1|_]) :- X =@= X1, X = X1.
alpha_member(X,[_|L]) :- alpha_member(X,L).

list_to_set_alpha_aux([X|L],M) :-
    alpha_member(X,L), !, list_to_set_alpha_aux(L,M).
list_to_set_alpha_aux([X|L],[X|M]) :- list_to_set_alpha_aux(L,M).
list_to_set_alpha_aux([],[]).

/* addition kb */
solver_setup:-
    qmpt0_refresh,
    qmpt0_add_fact(add(0, 0, 0)),
    qmpt0_add_fact(add(0, 1, 1)),
    qmpt0_add_fact(add(0, 2, 2)),
    qmpt0_add_fact(add(0, 3, 3)),
    qmpt0_add_fact(add(0, 4, 4)),
    qmpt0_add_fact(add(0, 5, 5)),
    qmpt0_add_fact(add(0, 6, 6)),
    qmpt0_add_fact(add(0, 7, 7)),
    qmpt0_add_fact(add(0, 8, 8)),
    qmpt0_add_fact(add(0, 9, 9)),
    qmpt0_add_fact(add(1, 0, 1)),
    qmpt0_add_fact(add(1, 1, 2)),
    qmpt0_add_fact(add(1, 2, 3)),
    qmpt0_add_fact(add(1, 3, 4)),
    qmpt0_add_fact(add(1, 4, 5)),
    qmpt0_add_fact(add(1, 5, 6)),
    qmpt0_add_fact(add(1, 6, 7)),
    qmpt0_add_fact(add(1, 7, 8)),
    qmpt0_add_fact(add(1, 8, 9)),
    qmpt0_add_fact(add(1, 9, 10)),
    qmpt0_add_fact(add(2, 0, 2)),
    qmpt0_add_fact(add(2, 1, 3)),
    qmpt0_add_fact(add(2, 2, 4)),
    qmpt0_add_fact(add(2, 3, 5)),
    qmpt0_add_fact(add(2, 4, 6)),
    qmpt0_add_fact(add(2, 5, 7)),
    qmpt0_add_fact(add(2, 6, 8)),
    qmpt0_add_fact(add(2, 7, 9)),
    qmpt0_add_fact(add(2, 8, 10)),
    qmpt0_add_fact(add(2, 9, 11)),
    qmpt0_add_fact(add(3, 0, 3)),
    qmpt0_add_fact(add(3, 1, 4)),
    qmpt0_add_fact(add(3, 2, 5)),
    qmpt0_add_fact(add(3, 3, 6)),
    qmpt0_add_fact(add(3, 4, 7)),
    qmpt0_add_fact(add(3, 5, 8)),
    qmpt0_add_fact(add(3, 6, 9)),
    qmpt0_add_fact(add(3, 7, 10)),
    qmpt0_add_fact(add(3, 8, 11)),
    qmpt0_add_fact(add(3, 9, 12)),
    qmpt0_add_fact(add(4, 0, 4)),
    qmpt0_add_fact(add(4, 1, 5)),
    qmpt0_add_fact(add(4, 2, 6)),
    qmpt0_add_fact(add(4, 3, 7)),
    qmpt0_add_fact(add(4, 4, 8)),
    qmpt0_add_fact(add(4, 5, 9)),
    qmpt0_add_fact(add(4, 6, 10)),
    qmpt0_add_fact(add(4, 7, 11)),
    qmpt0_add_fact(add(4, 8, 12)),
    qmpt0_add_fact(add(4, 9, 13)),
    qmpt0_add_fact(add(5, 0, 5)),
    qmpt0_add_fact(add(5, 1, 6)),
    qmpt0_add_fact(add(5, 2, 7)),
    qmpt0_add_fact(add(5, 3, 8)),
    qmpt0_add_fact(add(5, 4, 9)),
    qmpt0_add_fact(add(5, 5, 10)),
    qmpt0_add_fact(add(5, 6, 11)),
    qmpt0_add_fact(add(5, 7, 12)),
    qmpt0_add_fact(add(5, 8, 13)),
    qmpt0_add_fact(add(5, 9, 14)),
    qmpt0_add_fact(add(6, 0, 6)),
    qmpt0_add_fact(add(6, 1, 7)),
    qmpt0_add_fact(add(6, 2, 8)),
    qmpt0_add_fact(add(6, 3, 9)),
    qmpt0_add_fact(add(6, 4, 10)),
    qmpt0_add_fact(add(6, 5, 11)),
    qmpt0_add_fact(add(6, 6, 12)),
    qmpt0_add_fact(add(6, 7, 13)),
    qmpt0_add_fact(add(6, 8, 14)),
    qmpt0_add_fact(add(6, 9, 15)),
    qmpt0_add_fact(add(7, 0, 7)),
    qmpt0_add_fact(add(7, 1, 8)),
    qmpt0_add_fact(add(7, 2, 9)),
    qmpt0_add_fact(add(7, 3, 10)),
    qmpt0_add_fact(add(7, 4, 11)),
    qmpt0_add_fact(add(7, 5, 12)),
    qmpt0_add_fact(add(7, 6, 13)),
    qmpt0_add_fact(add(7, 7, 14)),
    qmpt0_add_fact(add(7, 8, 15)),
    qmpt0_add_fact(add(7, 9, 16)),
    qmpt0_add_fact(add(8, 0, 8)),
    qmpt0_add_fact(add(8, 1, 9)),
    qmpt0_add_fact(add(8, 2, 10)),
    qmpt0_add_fact(add(8, 3, 11)),
    qmpt0_add_fact(add(8, 4, 12)),
    qmpt0_add_fact(add(8, 5, 13)),
    qmpt0_add_fact(add(8, 6, 14)),
    qmpt0_add_fact(add(8, 7, 15)),
    qmpt0_add_fact(add(8, 8, 16)),
    qmpt0_add_fact(add(8, 9, 17)),
    qmpt0_add_fact(add(9, 0, 9)),
    qmpt0_add_fact(add(9, 1, 10)),
    qmpt0_add_fact(add(9, 2, 11)),
    qmpt0_add_fact(add(9, 3, 12)),
    qmpt0_add_fact(add(9, 4, 13)),
    qmpt0_add_fact(add(9, 5, 14)),
    qmpt0_add_fact(add(9, 6, 15)),
    qmpt0_add_fact(add(9, 7, 16)),
    qmpt0_add_fact(add(9, 8, 17)),
    qmpt0_add_fact(add(9, 9, 18)),
    /* artificial contradiction */
    qmpt0_add_fact(a),
    qmpt0_add_fact(-a),
    /*Logic forward rule*/
    qmpt0_add_rule(logic_forward([Z1, Z2], Res) <=
                [add(Z1, Z2, Res)]).
