function [vstate] = VState(N,theta)
    v0=[cos(theta);sin(theta)];
    vstate=1;
    for k=1:N
        vstate=kron(vstate,v0);
    end
    Nst=vstate'*vstate;
    vstate=vstate/sqrt(Nst); % init normalized
end

