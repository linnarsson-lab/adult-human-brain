% MixNB: a MATLAB object for maximum likelihood clustering with mixtures of sparse multivariate negative binomial distributions
% (C) Kenneth D. Harris, 2016
% kenneth.harris@ucl.ac.uk
% Released under the GPL

classdef MixNB
    properties
        nC; % number of cells (i.e. data points)
        nG; % number of genes (i.e. dimensions)
        nK; % number of clusters
        x; % expression vectors, nG by nC
        L; % log likelihoods, nC by nK
        Class; % assignment of best class for each cell, nC by 1. 
        SecondChoiceClass; % runner up class for each cell, nC by 1.
        BestL; % value of L of each cell in best class, nC by 1
        DeletionLoss; % how much you lose by deleting class, nK by 1
        prior; % class prior probabilities, nK by 1
        Active; % list of which genes are active in nb model
        mu; % mean of those genes (nK by nActive)
        L0; % log likelihood in single-cluster model, nC by 1
        Score; % gain in likelihood over single-cluster model, scalar
        Sim; % mean likelihood of each class in each other's model, nK by nK
        
        sx; % sum of expression in each class, nG by nK
        
        % parameters
        Verbose = 1; % 1 - intermediate 
        r = 2; % nb parameter r (scalar)
        RegN = 1e-4; % regularizeation for mean estimation, numerator
        RegD = 1; % regularization for estimating mean, denominator
        nActive = 150; % number of genes to keep in NB model
        nSplitTries = 10; % number of splits to try
        Tol = 1; % converge when score increases by less than this
        Gpu = 0;
        MaxIter = 100;
        AIC = 0; % multiple of AIC to use
        BIC = 1; % multiple of BIC to use
        ClassWorth;% penalty for each class
        
        % misc
        GeneName; % names of each gene, nG by 1 cell array of strings
        CellName; % name of each cell, nC by 1 cell array of strings
        ClassName; % names of each cluster, nK by 1 cell of strings
        ClusterTree; % output of linkage for clustering clusters
        OptimalClassOrder; % so neighbors are close
    end
    
    methods
        function m = MixNB(g, Scale, Class)
            % initialize from GeneSet g
            % expression is divided by Scale
            % Class should be a cell array of strings to start with
            % if empty, all are in one class.

            
            if nargin<2
                Scale = 1;
            end
            m.x = g.GeneExp/Scale;
            [m.nG, m.nC] = size(m.x);
            m.nK = 1;
            m.prior = 1;
            m.GeneName = g.GeneName;
            m.CellName = g.CellName;
            m.Score = 0;
            if m.Gpu
                m.x = gpuArray(m.x);
            end
            
            if nargin>=3
                [m.ClassName, ~, m.Class] = unique(Class);
            else
                m.Class = ones(m.nC,1);
                m.ClassName = {''};
            end
            
            % compute likelihood of each gene for all cells clustered together
            m.sx = sum(m.x,2); % total expression, nG by 1
            mu0 = (m.sx + m.RegN)/(m.nC + m.RegD); %  nG by 1
            p0 = mu0./(mu0+m.r); % nG by 1
            m.L0 = m.nC*m.r*log(1-p0) + m.sx.*log(p0); %  nG by 1
            
            % BIC penalty
            m.ClassWorth = m.BIC*m.nActive*log(m.nC)/2 + m.AIC*m.nActive;

        end
        
        function mOut = CellSubset(m, Cells)
            % takes a subset of the identified cell numbers
            mOut = m;
            
            mOut.nC = length(Cells);
            mOut.x = m.x(:,Cells);
%             mOut.L = m.L(Cells,:);
            mOut.Class = m.Class(Cells);
%             mOut.BestL = m.BestL(Cells);
            mOut.CellName = m.CellName(Cells);
            mOut.Score = nan; % it's invalid at this point
            mOut.Active = nan;
        end
        
        function mOut = PruneDeadClasses(m);
            % remove any classes with no members
            % uses m.prior to tell if a class is dead
            
            Alive = unique(m.Class);
            NewClass = nan(m.nK,1);
            NewClass(Alive) = (1:length(Alive)); % indexing array 
            
            mOut = m;
            mOut.nK = length(Alive);
            mOut.L = m.L(:, Alive);
            mOut.Class = NewClass(m.Class);
            mOut.SecondChoiceClass = NewClass(m.SecondChoiceClass);
            mOut.DeletionLoss = m.DeletionLoss(Alive);
            mOut.prior = m.prior(Alive);
            mOut.mu = m.mu(Alive,:);
            
            
            % you have to rerun ClusterClusters after this to get Sim and
            % ClassName
%             mOut.Sim = m.Sim(Alive, Alive);
            mOut.ClassName = m.ClassName(Alive);
        end
            
            
        
        function [m, Changes] = Estep(m)
            % [m, Changes] = Estep(m)
            %
            % compute m.Class from m.Active, m.mu, m.prior
            % Changes is a n by 3 array each row is [Cell, OldClass, NewClass]
            
            xSub = m.x(m.Active,:); % expression of active genes: nActive by nC

            % compute log likelihoods for each cell and cluster L(c,k)
            m.L = zeros(m.nC, m.nK); % nC by nK
            p = m.mu./(m.mu + m.r); % nK by nActive
            m.L = bsxfun(@plus, log(p)*xSub, sum(m.r*log(1-p), 2) + log(m.prior))';

            % find best class (and second best)
            [SortL, OrderL] = sort(m.L,2, 'descend');
            OldClass = m.Class;
            m.Class = OrderL(:,1);
            m.SecondChoiceClass = OrderL(:,2);
            m.BestL = SortL(:,1);
            
            % computed Changes
            Changed = find(OldClass ~= m.Class);
            Changes = [Changed, OldClass(Changed), m.Class(Changed)];
            
            % use second best to compute deletion penalty for each cluster
            if m.nK>1
                SecondL = SortL(:,2);
                m.DeletionLoss = accumarray(m.Class, m.BestL-SecondL);
            else 
                m.DeletionLoss = zeros(m.nK, m.nC);
            end
            
            % compute matrix giving mean likelihood of each class to each
            % other
            
            
%             m.Score = nan; % score is invalid until you run an M-step
        end
        
        function m = Mstep(m, Changes)
            % m = Mstep(m)
            % compute m.Active, m.mu, m.Score, m.prior from m.Class
            %
            % if second output Changes is provided, save time by using that
            % to more quickly compute m.sx from its previous value

            % compute sx: summed exp of each gene in each class, nG by nK
            if nargin<2
                m.sx = zeros(m.nG, m.nK);
                for k=1:m.nK
                    m.sx(:,k) = sum(m.x(:,m.Class==k),2);
                end
            else
                dsx = zeros(m.nG, m.nK);
                for k=1:m.nK
                    From = Changes(Changes(:,2)==k,1); % all cells leaving class k
                    To = Changes(Changes(:,3)==k,1); % all cells joining class k
                    
                    dsx(:,k) = sum(m.x(:,To),2) - sum(m.x(:,From),2);
                end            
                m.sx = m.sx + dsx;
            end
            
            n = accumarray(m.Class, 1, [m.nK 1]); % total cells in each class, 1 by nK 
            AllMu = bsxfun(@rdivide, m.sx+m.RegN, n'+m.RegD); %nG by nK
            AllP = AllMu ./ (AllMu + m.r); % nG by nK
            AllL = m.r*bsxfun(@times, n', log(1-AllP)) + m.sx.*log(AllP);

            % compute how much each gene benefits by clustering
            AlldL = bsxfun(@minus, sum(AllL,2), m.L0);

            [~, order] = sort(AlldL, 'descend');

            m.Active = order(1:m.nActive);
            m.mu = AllMu(m.Active,:)';
            m.Score = sum(AlldL(m.Active));
            
            % class probabilities
            m.prior = n/sum(n);
        end
        

        function [m, Score] = EM(m, MaxIter)
            % [m Score] = EM(m, MaxIter)
            % run an EM algorithm until convergence
            
            if nargin<2
                MaxIter = m.MaxIter;
            end
            
            m = m.Mstep(); % start with a full calculation
            LastScore = m.Score;
            OriginalScore = m.Score;
            OriginalClass = m.Class;
            if m.Verbose>=2
                fprintf('EM: %.0f -> ', m.Score);
            end
            
            
            for i=1:MaxIter
                [m, Changes] = m.Estep();
                if isempty(Changes); 
                    break; 
                end
                m = m.Mstep(Changes); % accelerated version
                
%                if m.Score<LastScore+m.Tol; break; end
                LastScore = m.Score;
            end
            
            m = m.Mstep; % last M-step just to clean everything up
            
            if m.Verbose > 2
                fprintf('%.0f in %d steps, changed %d, gain %.0f\n', ...
                    m.Score, i, sum(m.Class~=OriginalClass), LastScore-OriginalScore);
            end
        end

        
        function [Gains, SplitVal] = SuggestSplits(m,Subset)
            % [Gains, SplitPos] = BestSplits(w0)
            % sees how much you can improve the likelihood of hone gene by
            % splitting at any points. These are probably good starting
            % points for the EM algorithm
            %
            % input Subset is an index array that says use only these cells
            % output Gains is how much you gain for each gene (nG by 1)
            % output SplitPos (nG by 1) is the position of the split (<=this many)
            
            if m.Verbose; fprintf('Finding starting splits...'); end
            
            if nargin<2
                x = m.x'; % note internal x is nC by nG
                Subset = 1:m.nC;
            else
                x = m.x(:, Subset)';
            end
            
            xs = sort(x, 'ascend'); % sort each gene individually
%             clear x; % to save memory
            
            % cumulative sums for top and bottom halves of expression
            % - to evaluate splitting each gene in each position
            cx1 = cumsum(xs); % bottom half: nC by nG
            cx2 = bsxfun(@minus,cx1(end,:),cx1); % top half
            
            % mean expression for top and bottom halves
            %n1 = (1:m.nC)';
            n1 = (1:length(Subset))';
            n2 = n1(end)-n1;
            mu1 = bsxfun(@rdivide, cx1 + m.RegN, n1 + m.RegD);
            mu2 = bsxfun(@rdivide, cx2 + m.RegN, n2 + m.RegD);
            
            % nbin parameters
            p1 = mu1./(mu1 + m.r); % nC by nG
            p2 = mu2./(mu2 + m.r);
%             clear mu1 mu2;
            
            L1 = cx1.*log(p1) + m.r*bsxfun(@times, log(1-p1), n1);
            L2 = cx2.*log(p2) + m.r*bsxfun(@times, log(1-p2), n2);
            
            AlldL = bsxfun(@minus, L1+L2, L1(end,:));
            [Gains, SplitPos] = max(AlldL,[],1);
            SplitVal = xs(sub2ind(size(xs), SplitPos,1:m.nG));
            
            if m.Verbose; fprintf('Done\n'); end

        end
        
        function [BestClasses, BestSplitGene, BestScore] = EvaluateSplits(m, k, SplitGenes, SplitPoints)
            % [BestClasses, BestSplitGene, FinalScore] = EvaluateSplits(m, k, SplitGenes, SplitPoints)
            % evaluate how well it works to split cluster k 
            % by running EM for this cluster only
            %
            % SplitGenes: array of gene numbers to try splitting
            % SplitPoints: array of places to split them
            %
            % BestClasses: array of 1 or 2s, for cells of this class only
            % BestSplitGene: which one gave best score for this cluster
            % FinalScore: The score for the entire dataset after this split
            
            mSub = m.CellSubset(find(m.Class==k));
            mSub.Class = ones(mSub.nC,1);
            mSub = mSub.Mstep();
            OriginalScore = mSub.Score;
            
            if m.Verbose
                fprintf('Subclass score %.0f\n', OriginalScore);
            end
            
            BestScore = OriginalScore; % you need to have run M-step last for this to be accurate
            for i=1:length(SplitGenes);
                gene = SplitGenes(i);
                SplitPoint = SplitPoints(i);
                
                % create new structure with this split
                mTest = mSub;
                mTest.nK = 2;
                NewCells = (mTest.x(gene,:)>SplitPoint); % new cells have expression in split gene above split point
                mTest.Class(NewCells) = 2; % set their class
                mTest = mTest.Mstep(); % compute initial score
                
                if mSub.Verbose>=2
                    fprintf('Splitting %d cells by %s expression gains %.0f: ', ...
                        sum(NewCells), mSub.GeneName{gene}, mTest.Score - OriginalScore);
                end

                
                % now run EM algorithm
                mTest = mTest.EM();

                % was it the best?
                if mTest.Score>BestScore
                    BestScore = mTest.Score;
                    BestClasses = mTest.Class;
                    BestSplitGene = gene;
                end
            end
            
            if BestScore==mSub.Score
                if m.Verbose
                    fprintf(2, 'No splits beat the status quo');
                    BestClasses = ones(mSub.nC,1);
                    BestScore = OriginalScore;
                    BestSplitGene = '';
                    return;
                end
            end
            
            
            if m.Verbose
                fprintf(2, 'Best split was %s, final gain %.0f.\n', ...
                    m.GeneName{BestSplitGene}, BestScore-OriginalScore);
            end
%             mOut.mu = m2.mu;
%             mOut.Active = m2.Active;
%             mOut.Score = m2.Score;
%             mOut.prior = m2.prior;
            
        end

            
        
        function mOut = RecursiveSplit(m)
            % search for optimal split each of class in two
            % then split them all and run a full EM
            
            % need to reset BIC parameters because user might have changed
            % nActive
            m.ClassWorth = m.BIC*m.nActive*log(m.nC)/2 + m.AIC*m.nActive;
            
            BestSplitClass = 0;
            BestSplitGene = 0;
            OrigScore = m.Score;
            BestScore=0;
            mSplitAll = m;
            mSplitAll.nK = 2*m.nK;
            mSplitAll.Class = nan(mSplitAll.nC,1); % to catch any unassigned
            for k=1:m.nK;
                MyCells = find(m.Class==k);

                if length(MyCells)<2; % don't bother splitting 1 cell!
                    mSplitAll.Class(MyCells) = (k-1)*2 + 1;
                    continue; 
                end;
                if m.Verbose; 
                    fprintf(2,'\n\nTrying to split Class %s of %d cells. Full original score %.0f:\n', ...
                        m.ClassName{k}, sum(m.Class==k), m.Score);
                end
                
                % get split suggestions
                [Gains, SplitVal] = m.SuggestSplits(MyCells);

                % loop through all suggested splits as starting points
                [~, order] = sort(Gains, 'descend');
                [SubClasses, MyGene, MyScore] = EvaluateSplits(m, k, ...
                    order(1:m.nSplitTries), SplitVal(order(1:m.nSplitTries)));
                
                mSplitAll.Class(MyCells) = (k-1)*2 + SubClasses; % so split classes are neighbors
                mSplitAll.ClassName = cell(mSplitAll.nK,1); % give them blank names for now
                
            end
            
            mOut = mSplitAll.EM().PruneDeadClasses; % do a full EM run on it
            mOut = mOut.ClusterClusters(); % give them names

            fprintf(2, 'After full EM, gain of %.0f\n', mOut.Score-OrigScore);
            
            % now delete all classes not worth enough
            while 1
                [SmallestLoss, WorthlessClass] = min(mOut.DeletionLoss);
                if SmallestLoss>m.ClassWorth
                    break;
                end
                mOut = mOut.DeleteClass(WorthlessClass).PruneDeadClasses;
            end
               
            
            % have to run another M-step to avoid wrong names
            mOut = mOut.Mstep.ClusterClusters();
            
            if m.Verbose
                mOut.PrintClassInfo;
                figure(823746);
                clf;
                mOut.ExpMatrix('best');
                figure(823747);
                clf;
                mOut.PlotSim;

                drawnow; 
            end

            fprintf(2, 'Final gain of %.0f\n', mOut.Score-OrigScore);
            
        end
        
        function m = DeleteClass(m, k)
            % m = DeleteClass(k)
            %
            % assign all cells in class k to their second choice class
            % then run a full EM
            
            OriginalScore = m.Score;
            MyCells = (m.Class ==k);
            m.Class(MyCells) = m.SecondChoiceClass(MyCells);
            
            m = m.Mstep();
            ReassignScore = m.Score;
            if m.Verbose
                fprintf('Deleted class %s, lose %.0f\n', ...
                    m.ClassName{k}, OriginalScore-ReassignScore);
            end
            
            m = m.EM();
            
            if m.Verbose
                fprintf('Final loss %.0f\n', ...
                    OriginalScore-m.Score);
            end
        end
            
            
        
        function m = ClusterClusters(m)
            % performs hierarchical clustering on the clusters
            % then give them names
            
            % compute similarity of each cluster to each other by KL
            % distance
            m.Sim = zeros(m.nK, m.nK);
%             for k=1:m.nK
%                 m.Sim(k,:) = mean(bsxfun(@minus, m.L(m.Class==k,:), m.BestL(m.Class==k)),1);
%             end
            % replace all too-low values or nans with -ClassWorth
%            m.Sim(m.Sim<-m.ClassWorth | ~isfinite(m.Sim)) = -m.ClassWorth;
            
            % first compute likelihood of each cell to grand mean
            mu0 = (sum(m.x(m.Active,:),2)+m.RegN)/(m.nC + m.RegD); % grand mean expression of each gene
            p0 = mu0./(mu0+m.r); % nActive by 1;
            Lgrand = m.x(m.Active,:)'*log(p0) + sum(log(1-p0)*m.r);
            
            % now improvement over grand mean by using its own cluster
            for k=1:m.nK
                m.Sim(k,:) = mean(bsxfun(@minus, m.L(m.Class==k,:), Lgrand(m.Class==k)),1);
            end
            
            
            DistMat0 = -(m.Sim + m.Sim')/2;
            DistMat = bsxfun(@minus, DistMat0, diag(DistMat0));
            if any(DistMat(:)<0)
                warning('Distance matrix contains negative elements');
                DistMat(DistMat<0)=0;
            end
            m.ClusterTree = linkage(min(squareform(DistMat), m.ClassWorth), 'ward');
            
            Reg = 1;
            
            % to name clusters we want to make a cell array that says which children
            % belong to every tree node (should be build into matlab!)
            % node numbers <= nLeafs are leafs (i.e. classes), above are
            % joins
            nJoins = size(m.ClusterTree,1);
            nLeafs = m.nK;
            Children = cell(nJoins,1); % stores children of join j (i.e. node nLeafs+j)
            for j=1:nJoins
                LeftNode = m.ClusterTree(j,1);
                RightNode = m.ClusterTree(j,2);
                if LeftNode>nLeafs; 
                    LeftList = Children{LeftNode-nLeafs}; 
                else
                    LeftList = LeftNode; % it already was a child node
                end
                if RightNode>nLeafs; 
                    RightList = Children{RightNode-nLeafs}; 
                else
                    RightList = RightNode;
                end
                Children{j} = union(LeftList, RightList); % this stores child nodes

                % now get best names
                SumLeft = sum(m.sx(:,LeftList),2);
                SumRight = sum(m.sx(:,RightList),2);
                nLeft = sum(ismember(m.Class,LeftList));
                nRight = sum(ismember(m.Class, RightList));
                SumAll = sum(m.sx,2);
 
                [~, BestLeft] = sort(SumLeft./(SumRight + Reg*nRight), 'descend');
                [~, BestRight] = sort(SumRight./(SumLeft + Reg*nLeft), 'descend');

%                 [~, BestLeft] = sort(SumLeft./(SumAll + Reg*m.nC), 'descend');
%                 [~, BestRight] = sort(SumRight./(SumAll + Reg*m.nC), 'descend');
%                 
%                 MeanLeft = mean(m.x(:,LeftList),2);
%                 MeanRight = mean(m.x(:,RightList),2);
%                 MeanNotLeft = mean(m.x(:,setdiff(1:m.nC,LeftList)),2);
%                 MeanNotRight = mean(m.x(:,setdiff(1:m.nC,RightList)),2);
%                 
%                 [~, BestLeft] = sort((MeanLeft-MeanNotLeft)./...
%                     (MeanLeft + MeanNotLeft+ Reg*m.nC), 'descend');
%                 [~, BestRight] = sort((MeanRight-MeanNotRight)./...
%                     (MeanRight + MeanNotRight+ Reg*m.nC), 'descend');

                FirstDiff = find(BestLeft~=BestRight, 1, 'first');
                if ~isempty(FirstDiff)
                    LeftName{j} = m.GeneName{BestLeft(FirstDiff)};
                    RightName{j} = m.GeneName{BestRight(FirstDiff)};
                else
                    LeftName{j} = '1';
                    RightName{j} = '2';
                end
                
            end
                
            % now name all the nodes
            NodeName = cell(nJoins+nLeafs,1);
            NodeName{nJoins+nLeafs} = '';
            for j=nJoins:-1:1
                LeftNode = m.ClusterTree(j,1);
                RightNode = m.ClusterTree(j,2);
                NodeName{LeftNode}= [NodeName{j+nLeafs} '.' LeftName{j}];
                NodeName{RightNode}= [NodeName{j+nLeafs} '.' RightName{j}];
            end


            m.ClassName = NodeName(1:m.nK);
            
            % now arrange them in a nice order
            m.OptimalClassOrder = m.ClassName(optimalleaforder...
                (m.ClusterTree, squareform(DistMat)));

        end
        
        function PlotSim(m, Matrix)
            % PlotSim(Matrix)
            % plots a similarity matrix, by default m.Sim
            
            if nargin<2
                Matrix = m.Sim;
            end
            [~, order] = ismember(m.OptimalClassOrder, m.ClassName);
            SortNames = m.ClassName(order);
            imagesc(Matrix(order, order));
            colorbar;
            set(gca, 'xtick', 1:m.nK);
            set(gca, 'XTickLabel', SortNames);
            set(gca, 'XTickLabelRotation', 90);
            set(gca, 'ytick', 1:m.nK);
            set(gca, 'YTickLabel', SortNames);
            title('Cost for moving ...');
            ylabel('cells of this class...');
            xlabel('into this class');
        end
        
        function ExpMatrix(m, SortGenes)
            % ExpMatrix(SortOption)
            % option can be 'alpha' (alphabetical)
            % 'best' - sorts by class
            % default is what is in the structure, according to likelihood

            if nargin<2
                SortGenes = 'default';
            end
            
            [~, ClassOrder] = ismember(m.OptimalClassOrder, m.ClassName);
            
            if isequal(SortGenes,'best')
%                 GenePos = (1:m.nK)*(bsxfun(@rdivide, m.mu, sum(m.mu,1)));
                [~, GenePos] = max(m.mu(ClassOrder,:),[],1);
                [~, GeneOrder] = sort(GenePos);
            elseif isequal(SortGenes, 'alpha')
                [~, GeneOrder] = sort(m.GeneName(m.Active));
            else
                GeneOrder = 1:m.nActive;
            end
            clf
            
            imagesc(log10(1+m.mu(ClassOrder,GeneOrder)));
            set(gca, 'fontsize', 8);
            set(gca, 'xtick', 1:m.nActive, 'xticklabel', m.GeneName(m.Active(GeneOrder)), 'xticklabelrotation', 90);
            set(gca, 'ytick', 1:m.nK, 'yticklabel', m.ClassName(ClassOrder));
            grid on
            set(gca, 'GridColor', [1 1 1])
            
            % now put a second x-axis on top
            ax1=gca;
            ax2 = axes('Position', get(ax1, 'Position'),'Color', 'none');
            set(ax2, 'XAxisLocation', 'top','YAxisLocation','Right');
            % set the same Limits and Ticks on ax2 as on ax1;
            set(ax2, 'XLim', get(ax1, 'XLim'),'YLim', get(ax1, 'YLim'));
            set(ax2, 'XTick', get(ax1, 'XTick'), 'YTick', []); %get(ax1, 'YTick'));
            set(ax2, 'XTickLabel', get(ax1, 'XTickLabel'));%,'YTickLabel',get(ax1, 'YTickLabel'));
            set(ax2, 'xticklabelrotation', 90);
            
            colormap(hot);
            
        end
        
        function PrintClassInfo(m)
            nCells = accumarray(m.Class,1);
            [~, order] = ismember(m.OptimalClassOrder, m.ClassName);
            for k=order(:)'
                fprintf(2,'%40s: %3d cells, total value %6.0f, per cell %6.0f\n', ...
                    m.ClassName{k}, nCells(k), m.DeletionLoss(k), m.DeletionLoss(k)/nCells(k));
                
                % find top genes
                [~, gOrder] = sort(m.mu(k,:),2, 'descend');
                for i=gOrder(1:6)
                    fprintf('%s %.0f; ', m.GeneName{m.Active(i)}, m.mu(k, i));
                end
                fprintf('\n');
            end
        end
        
        function g2 = GeneSet(m, g);
            
            % g needs to have the same cell names as m, but not necessarily
            % in the same order
            [Sorted1 ,Order1] = sort(g.CellName);
            [Sorted2 ,Order2] = sort(m.CellName);
            if ~isequal(Sorted1, Sorted2)
db                error('IDs don''t match!');
            end
            
            NewOrder(Order1) = Order2; % inverse of one permutation times another
            
            CellCluster = m.Class;
            g2 = g;
            g2.Class = m.ClassName(CellCluster(NewOrder));
            g2.Class = g2.Class(:); % make it a column vector (I hate matlab)
            g2 = g2.SortByClass(m.OptimalClassOrder);
        end
    end
end
