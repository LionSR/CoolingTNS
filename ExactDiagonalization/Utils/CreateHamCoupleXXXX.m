function Ham = CreateHamCoupleXXXX(HamS,N,Delta,g,ind)
    sX = sparse([0 1; 1 0]);
    sZ = sparse([1 0; 0 -1]);
    sXXX = kron(kron(sX,sX),sX);
    Ham = kron(HamS,speye(2));
    Ham = Ham + Delta*kron(speye(2^(N)),sZ);
    ind = mod(ind,N-2)+1;
    Ham = Ham + g*kron(kron(speye(2^(ind-1)),kron(sXXX,speye(2^(N-ind-2)))),sX);
end