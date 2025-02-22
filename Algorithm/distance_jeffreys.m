function J = distance_jeffreys(P,Q)

J = 0.5 * ( trace(Q\P)  + trace(P\Q) ) -size(P,1); 

