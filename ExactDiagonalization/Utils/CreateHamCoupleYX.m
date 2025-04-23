function Ham = CreateHamCoupleYX(HamS,N,Delta,g,ind)
    sX = sparse([0 1; 1 0]);
    sY = sparse([0 -1i; 1i 0]);
    sZ = sparse([1 0; 0 -1]);
    Ham = kron(HamS,speye(2));
    Ham = Ham + Delta*kron(speye(2^(N)),sZ);
    Ham = Ham + g*kron(kron(speye(2^(ind-1)),kron(sY,speye(2^(N-ind)))),sX);
end