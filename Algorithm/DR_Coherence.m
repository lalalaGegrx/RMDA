function W=DR_Coherence(Coh,Label,dim_dr,iter)
    channels=size(Coh,1);
    if channels==2 || channels==dim_dr
        W=eye(channels);
    else
        spdDR_Obj=RMDA;
        spdDR_Obj.newDim=dim_dr;
        spdDR_Obj.nIter=iter;
        spdDR_Obj.trn_X=Coh;
        spdDR_Obj.trn_y=Label;
        spdDR_Obj.metric=3;

        W=spdDR_Obj.perform_graph_DA();
    end

end
