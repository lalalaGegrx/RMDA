function W=DR_PSD(psd,Label,dim_dr,iter)
    channels=size(psd,1);
    if channels==2 || channels==dim_dr
        W=eye(channels);
    else
        spdDR_Obj=RMDA_PSD;
        spdDR_Obj.newDim=dim_dr;
        spdDR_Obj.nIter=iter;
        spdDR_Obj.trn_X=psd;
        spdDR_Obj.trn_y=Label;
        spdDR_Obj.metric=3;

        W=spdDR_Obj.perform_graph_DA();
    end

end
