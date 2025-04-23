function [sZ,E,GSO,purity,popul] = Evolve(rho,UtAll,Vs,Niter,M,numelSchemes,rhozplus,Ncouples,sZa,N,HamS,psi_gs,Scheme,thetaind)
    E = zeros(sum(Niter),1);
    GSO = zeros(sum(Niter),1);
    purity = zeros(sum(Niter),1);
    popul = zeros(sum(Niter),M);
    sZ = zeros(sum(Niter),1);

    tot=0;
    for ind=1:numelSchemes
        tic;
        for i=1:Niter(ind)
            tot=tot+1;
            rho = kron(rho,rhozplus);
            Ut = squeeze(UtAll(ind,mod(i-1,Ncouples(ind))+1,:,:));
            rho = Ut*rho*Ut';
            sZ(tot) = real(trace(rho*sZa));
            rho = TrX(rho,2,[2^N 2]);
            E(tot) = real(trace(HamS*rho));
            GSO(tot) = real(psi_gs'*rho*psi_gs);
            purity(tot) = real(trace(rho^2));
            
            for j=1:M
                popul(tot,j) = real(Vs(:,j)'*rho*Vs(:,j));
            end
            
        end
        fprintf("\nthetaind=%d Scheme%d %s%d: E_cooled/N=%.4f, Purity=%.3f, GSpopulAll=%.3f, time = %0.2f sec",thetaind,ind,Scheme(ind),Niter(ind),E(tot)/N, purity(tot), GSO(tot), toc);
    end
end
