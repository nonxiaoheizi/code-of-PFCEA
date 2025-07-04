classdef MOEADFCHT < ALGORITHM
% <multi> <real> <constrained>
% MOEA/D with competitive multitasking
% nr    ---   2 --- Maximum number of solutions replaced by each offspring
 
%------------------------------- Copyright --------------------------------
% Copyright (c) 2024 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            [delta,nr] = Algorithm.ParameterSet(0.9,2);

             
            %% Generate the weight vectors
             [W,Problem.N] = UniformPoint(Problem.N,Problem.M);
             T = ceil(Problem.N/10);
            
            %% Detect the neighbours of each solution
             B = pdist2(W,W);
             [~,B] = sort(B,2);
             B = B(:,1:T);
            
            %% Initialization 
             Population{1} = Problem.Initialization();
             Z{1}   = min(Population{1}.objs,[],1);
             Conmin = min(overall_cv(Population{1}.cons));
             A      = Population{1}; 
            
            %% Optimization
             while Algorithm.NotTerminated(A) 
                CV    = sum(max(Population{1}.cons,0),2);
                fr    = length(find(CV<=0))/Problem.N;  
                sigma_obj = 0.3;
                sigma_cv = fr*10;
                Q = [];
                
                % For each solution
                for i = 1:Problem.N
                    % Choose the parents
                    if rand < delta
                        P = B(i,randperm(end));
                    else
                        P = randperm(Problem.N);
                    end  
                    % Generate an offspring

                    Offspring = OperatorDE(Problem,Population{1}(i),Population{1}(P(1)),Population{1}(P(2))); 
                    
                    % Update the ideal point
                    Z{1} = min(Z{1},Offspring.obj);
                    Conmin = min(Conmin,overall_cv(Offspring.con));
                    Zmax  = max([Population{1}.objs;Offspring.obj],[],1);
                    Conmax = max(overall_cv([Population{1}.cons;Offspring.con]));   
                    % Update the solutions in P by Tchebycheff approach
                    g_old = max(abs(Population{1}(P).objs-repmat(Z{1},length(P),1))./W(P,:),[],2);    
                    g_new = max(repmat(abs(Offspring.obj-Z{1}),length(P),1)./W(P,:),[],2);         
                    cv_old = overall_cv(Population{1}(P).cons);   
                    cv_new = overall_cv(Offspring.con);
                    if Conmax > Conmin
                        cv_old(cv_old > 0) = (cv_old(cv_old > 0)-Conmin)/(Conmax-Conmin);
                        cv_new(cv_new > 0) = (cv_new(cv_new > 0)-Conmin)/(Conmax-Conmin);    
                    end
                    new_old = LG(g_old - g_new,[sigma_obj 0]).*max(LG(cv_old - repmat(cv_new,length(P),1),[sigma_cv 0]),0.0001);   
                    old_new = LG(g_new - g_old,[sigma_obj 0]).*max(LG(repmat(cv_new,length(P),1) - cv_old,[sigma_cv 0]),0.0001);   
                    Population{1}(P(find(new_old>=old_new,nr))) = Offspring; 
                    Q = [Q Offspring];
                end
                if size(Q,2) > 0
                    s = size(A,2);
                    A =  archive([A Q],Problem.N);    % update Archive
                end 
             end
        end
    end  
end

function result = overall_cv(cv)
	cv(cv <= 0) = 0;cv = abs(cv);
	result = sum(cv,2);
end

function y = LG(x, para)
    if (x==0)
        y = repmat(0.5,length(x),1);
    else
        y = (1+exp(-1*(x-para(2))./repmat(para(1),length(x),1))).^(-1);
    end
end