function [Population,Next] = A_Update(Population,N)
%spea2
    %% Calculate the fitness of each solution
    Fitness = CalFitness(Population.objs,Population.cons);

    %% Environmental selection
    Next = Fitness < 1;            
    if sum(Next) < N
        [~,Rank] = sort(Fitness); 
        Next(Rank(1:N)) = true;
    elseif sum(Next) > N
        Del  = Truncation(Population(Next).objs,sum(Next)-N);    
        Temp = find(Next);     
        Next(Temp(Del)) = false;
    end
    % Population for next generation
    Population = Population(Next);
end

function Del = Truncation(PopObj,K)
% Select part of the solutions by truncation

    %% Truncation
    Distance = pdist2(PopObj,PopObj);
    Distance(logical(eye(length(Distance)))) = inf;
    Del = false(1,size(PopObj,1));
    while sum(Del) < K
        Remain   = find(~Del);
        Temp     = sort(Distance(Remain,Remain),2);
        [~,Rank] = sortrows(Temp);
        Del(Remain(Rank(1))) = true;
    end
end