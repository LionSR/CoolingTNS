function out = MultiSingleSpin(N,i,si)
    % Action of I_2 tensor I_2 tensor ... s_i tensor I_2 ... I_2 on a state psi
    
    sX = sparse([0 1; 1 0]);
    sY = sparse([0,-1i;1i,0]);
    sZ = sparse([1 0; 0 -1]);
    
    switch si
        case 1 % sx
            s = sX;
        case 2 % sy
            s = sY;
        case 3 % sz
            s = sZ;
    end
    
    IndL = 2^(i-1); % i-1 is the number of sites on the left.
    IndR = 2^(N-i); % on the right
    
    outT1= kron(speye(IndL), s);
    out= kron(outT1, speye(IndR));
end