classdef RMDA_PSD < handle
    %>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    properties
        %general parameters
        metric = 3; %1: AIRM, 2: Stein, 3: Jeffrey, 4: logEuclidean, 5: Euclidean
        nIter = 40;
        trn_X = [];
        trn_y = [];
        
        newDim = [];
        
        k_w = 3;    %within graph neighbor size for discriminant analysis
        k_b = 1;    %between graph neighbor size for discriminant analysis
        graph_lambda = 1;
        
        verbose = true;
    end
    %>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    properties (Access = private)
        nTrn = [];
        origDim = [];
        nClasses = [];        
        log_trn_X = [];
        aff_graph = [];
    end
    %>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    methods
        function W = perform_graph_DA(obj,metric)
            if (nargin == 2)
                obj.metric = metric;
            end
            obj.origDim = size(obj.trn_X,1);
            obj.nTrn = size(obj.trn_X,3);
            obj.nClasses = max(obj.trn_y);
            
            switch(obj.metric)
                case 1  %AIRM
                    W = obj.perform_AIRM_DA();
                case 2  %Stein
                    W = obj.perform_Stein_DA();
                case 3  %Jeffrey
                    W = obj.perform_Jeffrey_DA();
                case 4  %log-Euclidean
                    W = obj.perform_log_euc_DA();
                case 5  %Euclidean
                    W = obj.perform_euc_DA();
                otherwise
                    error('The metric is not defined');
            end
            
        end
    end
    
    %>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    methods(Access = private)
        
        %>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        function W = perform_AIRM_DA(obj)
            
            if (obj.verbose)
                fprintf('----------------------\n');
                fprintf('graph-based Discriminant Analysis using AIRM metric.\n');
                fprintf('Mapping from SPD(%d) -> SPD(%d)\n',obj.origDim,obj.newDim);
                fprintf('Number of training samples : %d.\n',obj.nTrn);
            end
            
            
            %generating the affinity function
            dist_orig = dist_AIRM(obj.trn_X);
            obj.aff_graph = obj.generate_Graphs(dist_orig);
            
            %initializing
            W0 = eye(obj.origDim,obj.newDim);
            
            import manopt.solvers.conjugategradient.*;
            import manopt.manifolds.grassmann.*;
            manifold = grassmannfactory(obj.origDim,obj.newDim);
            problem.M = manifold;
            problem.costgrad = @(W) graph_DA_CostGrad_AIRM(obj,W);
            %checkgradient(problem);
            
            [W, ~, problem_info] = conjugategradient(problem,W0,struct('maxiter',obj.nIter,'verbosity',3));
            if (obj.verbose)
                fprintf('----------------------\n\n');
            end
            
        end
      
        %>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        function W = perform_Stein_DA(obj)
            
            if (obj.verbose)
                fprintf('----------------------\n');
                fprintf('graph-based Discriminant Analysis using Stein metric.\n');
                fprintf('Mapping from SPD(%d) -> SPD(%d)\n',obj.origDim,obj.newDim);
                fprintf('Number of training samples : %d.\n',obj.nTrn);
            end
            
            
            %generating the affinity function
            dist_orig = dist_Stein(obj.trn_X);
            obj.aff_graph = obj.generate_Graphs(dist_orig);
            
            %initializing
            W0 = eye(obj.origDim,obj.newDim);
            
            import manopt.solvers.conjugategradient.*;
            import manopt.manifolds.grassmann.*;
            manifold = grassmannfactory(obj.origDim,obj.newDim);
            problem.M = manifold;
            problem.costgrad = @(W) graph_DA_CostGrad_Stein(obj,W);
            %             checkgradient(problem);
            
            [W, ~, problem_info] = conjugategradient(problem,W0,struct('maxiter',obj.nIter,'verbosity',3));
            if (obj.verbose)
                fprintf('----------------------\n\n');
            end
            
        end
        
        %>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        function W = perform_Jeffrey_DA(obj)
            
            if (obj.verbose)
                fprintf('----------------------\n');
                fprintf('graph-based Discriminant Analysis using Jeffrey metric.\n');
                fprintf('Mapping from SPD(%d) -> SPD(%d)\n',obj.origDim,obj.newDim);
                fprintf('Number of training samples : %d.\n',obj.nTrn);
            end
            
            
            %generating the affinity function
            dist_orig = dist_Jeffrey(obj.trn_X);
            obj.aff_graph = obj.generate_Graphs(dist_orig);
            
            %initializing
            W0 = eye(obj.origDim,obj.newDim);
            
            import manopt.solvers.conjugategradient.*;
            import manopt.manifolds.grassmann.*;
            manifold = grassmanncomplexfactory(obj.origDim,obj.newDim);
            problem.M = manifold;
            problem.costgrad = @(W) graph_DA_CostGrad_Jeffrey(obj,W);
            %checkgradient(problem);
            
            [W, ~, problem_info] = conjugategradient(problem,W0,struct('maxiter',obj.nIter,'verbosity',3));
            if (obj.verbose)
                fprintf('----------------------\n\n');
            end
            
        end       
        %>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        function W = perform_log_euc_DA(obj)
            
            if (obj.verbose)
                %fprintf('----------------------\n');
                %fprintf('graph-based Discriminant Analysis using log-Euclidean metric.\n');
                %fprintf('Mapping from SPD(%d) -> SPD(%d)\n',obj.origDim,obj.newDim);
                %fprintf('Number of training samples : %d.\n',obj.nTrn);
            end
            
            if (isempty(obj.log_trn_X))
                if (obj.verbose)
                    %fprintf('Preparing intermediate data.\n');
                end
                
                obj.log_trn_X = zeros(obj.origDim,obj.origDim,obj.nTrn);
                for tmpC1 = 1:obj.nTrn
                    obj.log_trn_X(:,:,tmpC1) = logm(obj.trn_X(:,:,tmpC1));
                end
                
                log_trn_mean = logeuclid_mean(obj.trn_X);
                
            end
                        
            %initializing
            W = eye(obj.origDim,obj.newDim);
            for tmpIter = 1:obj.nIter
                
                F_W = compute_F_W_log_Euc(obj,log_trn_mean,W);
                if (obj.verbose)
                    %computing the cost
                    cost0 = trace(W'*F_W*W);
                end
                
                [W,~] = eigs(F_W,obj.newDim,'sa');
                %computing the cost
                if (obj.verbose)
                    cost1 = trace(W'*F_W*W);
                    %fprintf('iter%d. Cost function before and after update %.3f -> %.3f.\n',tmpIter,cost0,cost1);
                end
            end %endfor iteration
            if (obj.verbose)
                %fprintf('----------------------\n\n');
            end
            
        end
        
        %>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        function W = perform_euc_DA(obj)
            if (obj.verbose)
                fprintf('----------------------\n');
                fprintf('graph-based Discriminant Analysis using Euclidean metric.\n');
                fprintf('Mapping from SPD(%d) -> SPD(%d)\n',obj.origDim,obj.newDim);
                fprintf('Number of training samples : %d.\n',obj.nTrn);
            end
            
            
            %generating the affinity function
            le_dist_orig  = dist_Euc(obj.trn_X);
            obj.aff_graph = obj.generate_Graphs(le_dist_orig);
            
            %initializing
            W = eye(obj.origDim,obj.newDim);
            for tmpIter = 1:obj.nIter
                
                F_W = obj.compute_F_W_Euc(obj.aff_graph,W);
                if (obj.verbose)
                    %computing the cost
                    cost0 = trace(W'*F_W*W);
                end
                
                [W,~] = eigs(F_W,obj.newDim,'sa');
                %computing the cost
                if (obj.verbose)
                    cost1 = trace(W'*F_W*W);
                    fprintf('iter%d. Cost function before and after update %.3f -> %.3f.\n',tmpIter,cost0,cost1);
                end
            end %endfor iteration
            if (obj.verbose)
                fprintf('----------------------\n\n');
            end
        end
        
        %>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        function a = generate_Graphs(obj,tmpDist)
            %Within Graph
            G_w = zeros(obj.nTrn);
            for tmpC1 = 1:obj.nTrn
                tmpIndex = find(obj.trn_y == obj.trn_y(tmpC1));
                [~,sortInx] = sort(tmpDist(tmpIndex,tmpC1));
                if (length(tmpIndex) < obj.k_w + 1)
                    max_w = length(tmpIndex);
                else
                    max_w = obj.k_w + 1;
                end
                G_w(tmpC1,tmpIndex(sortInx(1:max_w))) = 1;
            end
            %Between Graph
            G_b = zeros(obj.nTrn);
            for tmpC1 = 1:obj.nTrn
                tmpIndex = find(obj.trn_y ~= obj.trn_y(tmpC1));
                [~,sortInx] = sort(tmpDist(tmpIndex,tmpC1));
                if (length(tmpIndex) < obj.k_b)
                    max_b = length(tmpIndex);
                else
                    max_b = obj.k_b;
                end
                G_b(tmpC1,tmpIndex(sortInx(1:max_b))) = 1;
            end
            
            a = G_w - obj.graph_lambda*G_b;
        end
        %>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        function [outCost,outGrad] = graph_DA_CostGrad_AIRM(obj,W)
            I_m = eye(obj.newDim);
            WXW = zeros(obj.newDim,obj.newDim,obj.nTrn);
            iWXW = zeros(obj.newDim,obj.newDim,obj.nTrn);
            for tmpC1 = 1:obj.nTrn
                WXW(:,:,tmpC1) = W'*obj.trn_X(:,:,tmpC1)*W;
                iWXW(:,:,tmpC1) = I_m/WXW(:,:,tmpC1);
            end
            
            outCost = 0;
            dF = zeros(obj.origDim,obj.newDim);
            
            X_mean = riemann_mean(obj.trn_X);
            WX_meanW = (W'*X_mean*W);
            iWX_meanW = I_m/WX_meanW;
            W_iWX_meanW = W*iWX_meanW;
        
            %% We will look it back later 7.20
            for tmpC1 = 1:obj.nTrn
                outCost = outCost - dist_AIRM(WXW(:,:,tmpC1) , WX_meanW);
                X_i = obj.trn_X(:,:,tmpC1);                
                
                log_XY = logm(WXW(:,:,tmpC1)*iWX_meanW);              
                dF = dF - 4*((X_i*W)*iWXW(:,:,tmpC1) -(X_mean*W)*iWX_meanW)*log_XY;
                
            end
            outGrad = (eye(size(W,1)) - W*W')*dF;
        end
        
        %>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        function [outCost,outGrad] = graph_DA_CostGrad_Stein(obj,W)
            I_m = eye(obj.newDim);
            WXW = zeros(obj.newDim,obj.newDim,obj.nTrn);
            iWXW = zeros(obj.newDim,obj.newDim,obj.nTrn);
            for tmpC1 = 1:obj.nTrn
                WXW(:,:,tmpC1) = W'*obj.trn_X(:,:,tmpC1)*W;
                iWXW(:,:,tmpC1) = I_m/WXW(:,:,tmpC1);
            end
            
            outCost = 0;
            dF = zeros(obj.origDim,obj.newDim);
            
            [i,j,a_ij] = find(obj.aff_graph);
            
            %% We will look it back later
            a_ij = -ones(size(a_ij,1),1);
            
            for tmpC1 = 1:length(i)
                outCost = outCost + a_ij(tmpC1)*dist_Stein(WXW(:,:,i(tmpC1)) , WXW(:,:,j(tmpC1)));
                X_i = obj.trn_X(:,:,i(tmpC1));
                X_j = obj.trn_X(:,:,j(tmpC1));
                X_ij = 0.5*(X_i + X_j);
                dF = dF + a_ij(tmpC1)*(2*(X_ij*W)/(W'*X_ij*W)  ...
                    - (X_i*W)*iWXW(:,:,i(tmpC1)) - (X_j*W)*iWXW(:,:,j(tmpC1)));
                
            end
            outGrad = (eye(size(W,1)) - W*W')*dF;
        end
        
        %>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        function [outCost,outGrad] = graph_DA_CostGrad_Jeffrey(obj,W)
            I_m = eye(obj.newDim);
            Class_A_WXW = zeros(obj.newDim,obj.newDim,obj.nTrn);
            Class_A_iWXW = zeros(obj.newDim,obj.newDim,obj.nTrn);
            Class_A_X_sum = zeros(obj.origDim,obj.origDim);
            Class_A_iX_sum = zeros(obj.origDim,obj.origDim);

            outCost = 0;
            dF = zeros(obj.origDim,obj.newDim);
            
            %% Class A: Select data/transform data/inverse data     We implement it 3.5.2023
            %% Compute mean 
            Class_A_X = obj.trn_X(:,:,find(obj.trn_y == 0));
            for tmpC1 = 1:size(Class_A_X,3)

                Class_A_WXW(:,:,tmpC1) = W'*Class_A_X(:,:,tmpC1)*W;
                Class_A_iWXW(:,:,tmpC1) = I_m/Class_A_WXW(:,:,tmpC1);

                Class_A_X_sum = Class_A_X_sum + Class_A_X(:,:,tmpC1);
                Class_A_iX_sum = Class_A_iX_sum + eye(obj.origDim)/(Class_A_X(:,:,tmpC1));

            end

            middle_term = (Class_A_iX_sum^(0.5)*Class_A_X_sum*Class_A_iX_sum^(0.5))^(0.5);
            Class_A_X_mean = Class_A_iX_sum^(-0.5)*middle_term*Class_A_iX_sum^(-0.5);

            Class_A_WX_meanW = (W'*Class_A_X_mean*W);
            Class_A_iWX_meanW = I_m/Class_A_WX_meanW;
 
            %% Class B: Select data/transform data/inverse data     We implement it 3.5.2023
            %% Compute mean
            Class_B_WXW = zeros(obj.newDim,obj.newDim,obj.nTrn);
            Class_B_iWXW = zeros(obj.newDim,obj.newDim,obj.nTrn);
            Class_B_X_sum = zeros(obj.origDim,obj.origDim);
            Class_B_iX_sum = zeros(obj.origDim,obj.origDim);

            Class_B_X = obj.trn_X(:,:,find(obj.trn_y == 1));
            for tmpC1 = 1:size(Class_B_X,3)

                Class_B_WXW(:,:,tmpC1) = W'*Class_B_X(:,:,tmpC1)*W;
                Class_B_iWXW(:,:,tmpC1) = I_m/Class_B_WXW(:,:,tmpC1);

                Class_B_X_sum = Class_B_X_sum + Class_B_X(:,:,tmpC1);
                Class_B_iX_sum = Class_B_iX_sum + eye(obj.origDim)/(Class_B_X(:,:,tmpC1));

            end

            middle_term = (Class_B_iX_sum^(0.5)*Class_B_X_sum*Class_B_iX_sum^(0.5))^(0.5);
            Class_B_X_mean = Class_B_iX_sum^(-0.5)*middle_term*Class_B_iX_sum^(-0.5);

            Class_B_WX_meanW = (W'*Class_B_X_mean*W);
            Class_B_iWX_meanW = I_m/Class_B_WX_meanW;

      
            %% We start to compute loss
            loss_1 = 0;
            loss_2 = 0;
            loss_3 = 0;
            dF_1 = zeros(obj.origDim,obj.newDim);
            dF_2 = zeros(obj.origDim,obj.newDim);
            dF_3 = zeros(obj.origDim,obj.newDim);

            % Compute loss 1
            for tmpC1 = 1:size(Class_A_X,3)

                loss_1 = loss_1 + dist_Jeffrey(Class_A_WXW(:,:,tmpC1) , Class_A_WX_meanW);   

                X_i = Class_A_X(:,:,tmpC1);
                
                term_1 = X_i*W*(Class_A_iWX_meanW - Class_A_iWXW(:,:,tmpC1) * Class_A_WX_meanW * Class_A_iWXW(:,:,tmpC1));
                term_2 = Class_A_X_mean*W*(Class_A_iWXW(:,:,tmpC1) - Class_A_iWX_meanW*Class_A_WXW(:,:,tmpC1)*Class_A_iWX_meanW);

                dF_1 = dF_1 + term_1 + term_2;
                
            end
            % Compute loss 2
            for tmpC1 = 1:size(Class_B_X,3)
                
                loss_2 = loss_2 + dist_Jeffrey(Class_B_WXW(:,:,tmpC1) , Class_B_WX_meanW);
                                
                X_i = Class_B_X(:,:,tmpC1);
                
                term_1 = X_i*W*(Class_B_iWX_meanW - Class_B_iWXW(:,:,tmpC1) * Class_B_WX_meanW * Class_B_iWXW(:,:,tmpC1));
                term_2 = Class_B_X_mean*W*(Class_B_iWXW(:,:,tmpC1) - Class_B_iWX_meanW*Class_B_WXW(:,:,tmpC1)*Class_B_iWX_meanW);

                dF_2 = dF_2 + term_1 + term_2;
                
            end
            % Compute loss 3

            loss_3 = loss_3 + dist_Jeffrey(Class_A_WX_meanW , Class_B_WX_meanW);

            term_1 = Class_A_X_mean*W*(Class_B_iWX_meanW - Class_A_iWX_meanW * Class_B_WX_meanW * Class_A_iWX_meanW);
            term_2 = Class_B_X_mean*W*(Class_A_iWX_meanW - Class_B_iWX_meanW*Class_A_WX_meanW*Class_B_iWX_meanW);


            dF_3 = dF_3 + term_1 + term_2;
            

            outCost = loss_1 + loss_2 - loss_3 + 0.02*norm(W,'fro')^2;
            dF = dF_1 + dF_2 - dF_3;
            % dF = dF + 0.02*W;

            outGrad = (eye(size(W,1)) - W*W')*dF+0.02*2*W;
        end
        
        %>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        function WXW = map_trn_X(obj,W)
            WXW = zeros(obj.newDim,obj.newDim,obj.nTrn);
            for tmpC1 = 1:obj.nTrn
                WXW(:,:,tmpC1) = W'*obj.trn_X(:,:,tmpC1)*W;
            end
        end
        
        %>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        function F_W = compute_F_W_log_Euc(obj,log_trn_mean,W)
            F_W = zeros(obj.origDim,obj.origDim);
            WW = W*W';
            for tmpC1 = 1:obj.nTrn
                diff = obj.log_trn_X(:,:,tmpC1) - log_trn_mean;
                F_W = F_W - diff*WW*diff;
            end
            %F_W = obj.symm(F_W);
        end
        
        %>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        function F_W = compute_F_W_Euc(obj,a,W)
            F_W = zeros(obj.origDim,obj.origDim);
            a_sym = a + a';
            a_sym(1:obj.nTrn+1:end) = 0;
            [i,j,a_ij] = find(a_sym);
            WW = W*W';
            for tmpC1 = 1:length(i)
                diff = obj.trn_X(:,:,i(tmpC1)) - obj.trn_X(:,:,j(tmpC1));
                F_W = F_W + a_ij(tmpC1)*diff*WW*diff;
            end
            F_W = obj.symm(F_W);
        end
        
        %>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        function D = distEucVec(~, X, Y )
            Yt = Y';
            XX = sum(X.*X,2);
            YY = sum(Yt.*Yt,1);
            D = bsxfun(@plus,XX,YY) - 2*X*Yt;
        end

        
        %>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        function sym_X = symm(~,X)
            sym_X = .5*(X + X');
        end
        
    end
end