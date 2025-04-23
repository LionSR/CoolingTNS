function H = CreateHamZZXZ(N,J,hx,hz)
    sX = sparse([0 1; 1 0]);
    sZ = sparse([1 0; 0 -1]);
    sZZ = kron(sZ,sZ);
    H = hx*kron(sX,speye(2^(N-1)));
    for i=2:N
        H = H+hx*kron(speye(2^(i-1)),kron(sX,speye(2^(N-i))));
    end
    for i=1:N
        H = H+hz*kron(speye(2^(i-1)),kron(sZ,speye(2^(N-i))));
    end
    for i=1:N-1
        H = H+J*kron(speye(2^(i-1)),kron(sZZ,speye(2^(N-1-i))));
    end    
end