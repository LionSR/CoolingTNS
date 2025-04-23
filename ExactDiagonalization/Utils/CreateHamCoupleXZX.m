function Ham = CreateHamCoupleXZX(HamS,N,Delta,g,ind)
    sX = sparse([0 1; 1 0]);
    sZ = sparse([1 0; 0 -1]);
    sZX = kron(sZ,sX);
    Ham = kron(HamS,speye(2));
    Ham = Ham + Delta*kron(speye(2^(N)),sZ);
    ind = mod(ind,N-1)+1;
    Ham = Ham + g*kron(kron(speye(2^(ind-1)),kron(sZX,speye(2^(N-ind-1)))),sX);
end