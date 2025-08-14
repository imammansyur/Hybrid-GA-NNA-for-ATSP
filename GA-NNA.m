function GA_with_NN_and_TSPLIB()
    % Set whether to use TSPLIB ATSP dataset or real-world data
    useTSPLIB = false;
    tsplibFolder = 'ALL_atsp';

    % Load distance matrix and metadata from data files
    if useTSPLIB
        tsplibFiles = dir(fullfile(tsplibFolder, '*.atsp'));
        dataList = {tsplibFiles.name};
    else
        distMatrix = readmatrix('distance-matrix.csv');
        places = readtable('places.csv');
        dataList = unique(places.wilayah);
    end

    % Prepare CSV output file
    timestamp = string(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
    subfolderPath = fullfile('outputs', timestamp+'-GA+NN');
    if ~exist(subfolderPath, 'dir')
        mkdir(subfolderPath);
    end
    csvFile = fullfile(subfolderPath, 'results_' + timestamp + '-GA+NN.csv');
    
    % Write CSV header
    header = {'Data', 'NN_Distance', 'GA_Distance', 'Runtime', ...
              'GA_Population', 'GA_Convergence', 'GA_Generations', 'GA_MaxGenerations', 'GA_CrossoverRate', 'GA_MutationRate', 'GA_EliteCount', ...
              'NN_Route', 'GA_Route'};
    writecell(header, csvFile);

    %% Loop through each data (wilayah)
    for w = 1:length(dataList)
        dataName = dataList{w};
        fprintf('\nProcessing Data: %s\n', dataName);

        if useTSPLIB
            tsplibFile = fullfile(tsplibFolder, dataName);
            [distMatrix, nodeNames] = readTSPLIB_ATSP(tsplibFile);
            places = table(nodeNames', 'VariableNames', {'name'});
            places.wilayah = repmat(dataName, height(places), 1);
            places.latitude = zeros(height(places),1); % Placeholder
            places.longitude = zeros(height(places),1); % Placeholder
            
            dataIdx = (1:height(places))';
        else
            % Filter rows to current data (wilayah)
            dataIdx = find(strcmp(places.wilayah, dataName));
            % Ensure depot (point 1) is included 
            if ~ismember(1, dataIdx)
                dataIdx = [1; dataIdx];
            end
        end

        % Skip if not enough points
        if numel(unique(dataIdx)) < 2
            warning("Skipping wilayah %s: not enough valid points", dataName);
            continue;
        end

        % Extract distance matrix and metadata from the current data
        localDist = distMatrix(dataIdx, dataIdx);
        localPlaces = places(dataIdx, :);

        %% Run GA & NN TSP Solver
        tic;
        [bestRoute, bestDist, nnRoute, nnDist, bestFitnessOverTime, GAParameters] = solverTSP_GA_with_NN(localDist);
        runtime = toc;

        % Add depot to the start & end of route
        fullGARoute = [1, bestRoute, 1];
        fullNNRoute = [1, nnRoute, 1];

        %% Print results
        disp('GA_Parameters');
        disp(GAParameters);
        fprintf('Initial NN Distance: %.2f\n', nnDist);
        disp(strjoin(string(dataIdx(fullNNRoute)), '-'));
        fprintf('\nFinal GA Distance: %.2f\n', bestDist);
        disp(strjoin(string(dataIdx(fullGARoute)), '-'));
        fprintf('\nRuntime: %.2f seconds\n', runtime);

        % Save to CSV
        row = {
            dataName, nnDist, bestDist, runtime, ...
            GAParameters.populationSize, GAParameters.gen - GAParameters.stallCounter, ...
            GAParameters.gen, GAParameters.numGenerations, GAParameters.crossoverRate, ...
            GAParameters.mutationRate, GAParameters.eliteCount, ...
            strjoin(string(dataIdx(fullNNRoute)), '-'), ...
            strjoin(string(dataIdx(fullGARoute)), '-')
        };
        writecell(row, csvFile, 'WriteMode', 'append');

        % Plotting, doesn't plot TSPLIB route because no coordinate
        if ~useTSPLIB
            plotRoutes(localPlaces, fullNNRoute, fullGARoute, dataName, subfolderPath, timestamp);
        end
        plotConvergence(bestFitnessOverTime, dataName, subfolderPath, timestamp);
    end
end

%% GA + NN Solver Function
function [bestRoute, bestDist, nnRoute, nnDist, bestFitnessOverTime, params] = solverTSP_GA_with_NN(distMatrix)
    numCities = size(distMatrix, 1);
    depot = 1;
    cities = 2:numCities;

    % GA Parameters
    %params.populationSize = numCities^2;
    %params.numGenerations = numCities*5;
    %% params.mutationRate = min(0.3, 4 / numCities);
    %% params.eliteCount = max(2, round(0.02 * params.populationSize));
    %% params.populationSize = round(10 * log(numCities) * numCities);
    %% params.numGenerations = round(20 * log(numCities) * numCities);
    %params.mutationRate = max(0.02, min(0.3, 0.5 / sqrt(numCities)));
    %params.eliteCount = max(2, round(0.02 * params.populationSize));
    
    params.populationSize = 400;
    params.numGenerations = 500;
    params.crossoverRate = 0.9;
    params.mutationRate = 0.01;
    params.eliteCount = max(2, round(0.1 * params.populationSize));
    
    % Stall limit setting
    %stallLimit = round(0.3 * params.numGenerations);
    stallLimit = params.numGenerations;
    params.stallCounter = 0;
    lastBestFitness = Inf;

    % NN Initialization
    nnRoute = nearestNeighborRoute(distMatrix);
    nnDist = calculateRouteDistance([depot, nnRoute, depot], distMatrix);

    % Initialize population
    population = zeros(params.populationSize, numel(cities));
    population(1,:) = nnRoute;
    for i = 2:params.populationSize
        population(i,:) = cities(randperm(numel(cities)));
    end

    bestFitnessOverTime = zeros(params.numGenerations, 1);

    params.gen = 1;
    while params.gen <= params.numGenerations
        fitness = arrayfun(@(i) calculateRouteDistance([depot, population(i,:), depot], distMatrix), 1:params.populationSize)';
        bestFitnessOverTime(params.gen) = min(fitness);
    
        if bestFitnessOverTime(params.gen) < lastBestFitness - 1e-6
            params.stallCounter = 0;
            lastBestFitness = bestFitnessOverTime(params.gen);
        else
            params.stallCounter = params.stallCounter + 1;
        end
    
        if params.stallCounter >= stallLimit
            fprintf('Early stopping at generation %d due to no improvement after %d generations.\n', params.gen, stallLimit);
            break;
        end
    
        [~, idx] = sort(fitness);
        newPopulation = population(idx(1:params.eliteCount), :);
    
        while size(newPopulation,1) < params.populationSize
            p1 = tournamentSelect(population, fitness, 3);
            p2 = tournamentSelect(population, fitness, 3);
            
            if rand < params.crossoverRate
                child = orderCrossover(p1, p2);
            else
                child = p1;
            end

            if rand < params.mutationRate
                child = mutateRoute(child);
            end

            newPopulation(end+1, :) = child;
        end
    
        population = newPopulation;
        params.gen = params.gen + 1;
    end
    
    bestFitnessOverTime = bestFitnessOverTime(1:params.gen-1);

    fitness = arrayfun(@(i) calculateRouteDistance([depot, population(i,:), depot], distMatrix), 1:params.populationSize)';
    [~, bestIdx] = min(fitness);
    bestRoute = population(bestIdx, :);
    bestDist = fitness(bestIdx);
end

%% Utilities
function d = calculateRouteDistance(route, distMatrix)
    d = sum(arrayfun(@(i) distMatrix(route(i), route(i+1)), 1:length(route)-1));
end

function child = orderCrossover(p1, p2)
    n = numel(p1);
    idx = sort(randperm(n,2));
    child = nan(1,n);
    child(idx(1):idx(2)) = p1(idx(1):idx(2));
    child(isnan(child)) = p2(~ismember(p2, child));
end

function mutated = mutateRoute(route)
    idx = randperm(numel(route), 2);
    mutated = route;
    mutated(idx) = mutated(fliplr(idx));
end

function winner = tournamentSelect(pop, fitness, k)
    idx = randperm(size(pop,1), k);
    [~, best] = min(fitness(idx));
    winner = pop(idx(best), :);
end

function route = nearestNeighborRoute(distMatrix)
    n = size(distMatrix,1);
    visited = false(1,n);
    route = [];
    current = 1;
    visited(current) = true;

    while any(~visited)
        unvisited = find(~visited);
        [~, idx] = min(distMatrix(current, unvisited));
        next = unvisited(idx);
        route(end+1) = next;
        visited(next) = true;
        current = next;
    end

    route = route(route ~= 1);  % Remove depot
end

function plotRoutes(places, nnRoute, gaRoute, wilayahName, subfolderPath, timestamp)
    figure;
    geoplot(places.latitude(nnRoute), places.longitude(nnRoute), 'b--o', 'DisplayName', 'Nearest Neighbor');
    hold on;
    geoplot(places.latitude(gaRoute), places.longitude(gaRoute), 'r-o', 'LineWidth', 2, 'DisplayName', 'Genetic Algorithm');
    legend;
    title(['Route Comparison - ', wilayahName]);
    geobasemap streets;
    routeFile = fullfile(subfolderPath, sprintf('route_%s_%s.png', wilayahName, timestamp));
    saveas(gcf, routeFile);
end

function plotConvergence(fitness, wilayahName, subfolderPath, timestamp)
    figure;
    plot(1:length(fitness), fitness, 'LineWidth', 2);
    title(['GA+NN Convergence - ', wilayahName]);
    xlabel('Generation'); ylabel('Best Distance');
    grid on;
    convergenceFile = fullfile(subfolderPath, sprintf('convergence_%s_%s.png', wilayahName, timestamp));
    saveas(gcf, convergenceFile)
end

function [distMatrix, names] = readTSPLIB_ATSP(file)
    fid = fopen(file, 'r');
    names = {}; data = [];
    while true
        line = fgetl(fid);
        if contains(line, 'EDGE_WEIGHT_SECTION')
            break;
        end
    end
    buffer = [];
    while true
        line = fgetl(fid);
        if contains(line, 'EOF') || ~ischar(line)
            break;
        end
        nums = sscanf(line, '%f');
        buffer = [buffer; nums];
    end
    dim = sqrt(length(buffer));
    distMatrix = reshape(buffer, dim, dim);
    names = string(1:dim);
    fclose(fid);
end
