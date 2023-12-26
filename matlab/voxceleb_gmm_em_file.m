function gmm_all = voxceleb_gmm_em_file(file_path, file_list, nmix, ds_factor, nworkers, quiet)
% fits a nmix-component Gaussian mixture model (GMM) to data in dataList
% using niter EM iterations per binary split. The process can be
% parallelized in nworkers batches using parfor.
%
% Inputs:
%   - dataList    : ASCII file containing feature file names (1 file per line) 
%					or a cell array containing features (nDim x nFrames). 
%					Feature files must be in uncompressed HTK format.
%   - nmix        : number of Gaussian components (must be a power of 2)
%   - final_iter  : number of EM iterations in the final split
%   - ds_factor   : feature sub-sampling factor (every ds_factor frame)
%   - nworkers    : number of parallel workers
%   - gmmFilename : output GMM file name (optional)
%
% Outputs:
%   - gmm		  : a structure containing the GMM hyperparameters
%					(gmm.mu: means, gmm.sigma: covariances, gmm.w: weights)
%
%
% Omid Sadjadi <s.omid.sadjadi@gmail.com>
% Microsoft Research, Conversational Systems Research Center

gmm_all = {};


if ( nargin <= 4 )
	ds_factor = 1;
end
if ( nargin <= 5 )
	nworkers = feature('numCores');
end
if ( nargin <= 6 )
	quiet = 0;
end

if ischar(nmix), nmix = str2double(nmix); end
% if ischar(final_niter), final_niter = str2double(final_niter); end
if ischar(ds_factor), ds_factor = str2double(ds_factor); end
if ischar(nworkers), nworkers = str2double(nworkers); end

[ispow2, ~] = log2(nmix);
if ( ispow2 ~= 0.5 )
	error('oh dear! nmix should be a power of two!');
end

% if ischar(dataList) || isstring(dataList) || iscellstr(dataList)
% 	dataList = load_data(dataList);
% end
% if ~iscell(dataList)
% 	error('Oops! dataList should be a cell array!');
% end

nfiles = length(file_list);

% fprintf('\n\nInitializing the GMM hyperparameters ...\n');
if ~quiet, show_message('Initializing the GMM hyperparameters ...'); end
[gm, gv] = comp_gm_gv(file_path, file_list);
gmm = gmm_init(gm, gv); 

% gmm_all{1} = gmm;

% gradually increase the number of iterations per binary split
% mix = [1 2 4 8 16 32 64 128 256 512 1024];
% niter = [1 2 4 4  4  4  6  6   10  10  15];
niter = [2 10 20 20  20  30  30  30   30  30  30];

% niter(log2(nmix) + 1) = final_niter;

mix = 1;
while ( mix <= nmix )
	if ( mix >= nmix/2 ), ds_factor = 1; end % not for the last two splits!
%     fprintf('\nRe-estimating the GMM hyperparameters for %d components ...\n', mix);
    if ~quiet, show_message(['Re-estimating the GMM hyperparameters for ', num2str(mix),' components ...'] ); end
    
%     iternum = 30;
%     if (mix == 1) 
%         iternum = 1;
%     end
    for iter = 1 : niter(log2(mix) + 1)
%         fprintf('EM iter#: %d \t', iter);
        if ~quiet, fprintf('EM iter#: %d \t', iter); end
        N = 0; 
        F = 0; 
        S = 0; 
        
        L = 0; 
        nframes = 0;
        tim = tic;
        parfor (idx = 1 : nfiles, nworkers)
%         for idx = 1 : nfiles
            filename = fullfile(file_path, [file_list{idx}, '.h5']);
            data = load_one_file(filename);
%             data = double(data);
            [n, f, s, l] = expectation(data, gmm);
            N = N + n; 
            F = F + f; 
            S = S + s; 
            
            L = L + sum(l);
			nframes = nframes + length(l);
        end
        tim = toc(tim);
%         fprintf('[llk = %.2f] \t [elaps = %.2f s]\n', L/nframes, tim);
        if ~quiet, fprintf('[llk = %.4f] \t [elaps = %.2f s]\n', L/nframes, tim); end
        gmm = maximization(N, F, S);
    end
    
    gmm_all{end + 1} = gmm;
    
    if ( mix < nmix )
        gmm = gmm_mixup(gmm); 
    end
    mix = mix * 2;
end

end


function [gm, gv] = comp_gm_gv(file_path, file_list)
    nframes = 0;
    gm = 0;
    gv = 0;
    
    parfor idx = 1 : length(file_list)
        filename = fullfile(file_path, [file_list{idx}, '.h5']);
        data = load_one_file(filename);
%         data = double(data);
        
        nframes = nframes + size(data, 2);
        gm = gm + sum(data, 2);
        gv = gv + sum(data .* data, 2);
        
    end
    gm = gm / nframes;
    gv = gv / nframes - gm .* gm;

end


function gmm = gmm_init(glob_mu, glob_sigma)
    % initialize the GMM hyperparameters (Mu, Sigma, and W)
    gmm.mu    = glob_mu;
    gmm.sigma = glob_sigma;
    gmm.w     = 1;
end

function [N, F, S, llk] = expectation(data, gmm)
    % compute the sufficient statistics
    [post, llk] = postprob(data, gmm.mu, gmm.sigma, gmm.w(:));
    N = sum(post, 2)';
    F = data * post';
    S = (data .* data) * post';
end

function [post, llk] = postprob(data, mu, sigma, w)
    % compute the posterior probability of mixtures for each frame
    post = lgmmprob(data, mu, sigma, w);
    llk  = logsumexp(post, 1);
    post = exp(bsxfun(@minus, post, llk));
end

function logprob = lgmmprob(data, mu, sigma, w)
    % compute the log probability of observations given the GMM
    ndim = size(data, 1);
    C = sum(mu.*mu./sigma) + sum(log(sigma));
    D = (1./sigma)' * (data .* data) - 2 * (mu./sigma)' * data  + ndim * log(2 * pi);
    logprob = -0.5 * (bsxfun(@plus, C',  D));
    logprob = bsxfun(@plus, logprob, log(w));
end

function y = logsumexp(x, dim)
    % compute log(sum(exp(x),dim)) while avoiding numerical underflow
    xmax = max(x, [], dim);
    y    = xmax + log(sum(exp(bsxfun(@minus, x, xmax)), dim));
    ind  = find(~isfinite(xmax));
    if ~isempty(ind)
        y(ind) = xmax(ind);
    end
end

function gmm = maximization(N, F, S)
    % ML re-estimation of GMM hyperparameters which are updated from accumulators
    w  = N / sum(N);
    mu = bsxfun(@rdivide, F, N);
    sigma = bsxfun(@rdivide, S, N) - (mu .* mu);
    sigma = apply_var_floors(w, sigma, 0.1);
    gmm.w = w;
    gmm.mu= mu;
    gmm.sigma = sigma;
end

function sigma = apply_var_floors(w, sigma, floor_const)
    % set a floor on covariances based on a weighted average of component
    % variances
    vFloor = sigma * w' * floor_const;
    sigma  = bsxfun(@max, sigma, vFloor);
    % sigma = bsxfun(@plus, sigma, 1e-6 * ones(size(sigma, 1), 1));
end

% function gmm = gmm_mixup(gmm)
%     % perform a binary split of the GMM hyperparameters
%     mu = gmm.mu; sigma = gmm.sigma; w = gmm.w;
%     [ndim, nmix] = size(sigma);
%     [sig_max, arg_max] = max(sigma);
%     eps = sparse(0 * mu);
%     eps(sub2ind([ndim, nmix], arg_max, 1 : nmix)) = sqrt(sig_max);
%     % only perturb means associated with the max std along each dim 
%     new_mu2 = [mu - eps, mu + eps];
%     % mu = [mu - 0.2 * eps, mu + 0.2 * eps]; % HTK style
%     new_sigma2 = [sigma, sigma];
%     new_w2 = [w, w] * 0.5;
%     
%     new_w = zeros(size(new_w2));
%     new_mu = zeros(size(new_mu2));
%     new_sigma = zeros(size(new_sigma2));
%     
%     for i = 1 : nmix
%         new_w(2*i-1) = w(i) * 0.5;
%         new_w(2*i)   = w(i) * 0.5;
%         new_sigma(:, 2*i-1) = sigma(:, i);
%         new_sigma(:, 2*i)   = sigma(:, i);
%         
%         new_mu(:, 2*i-1) = mu(:, i) - eps(:, i);
%         new_mu(:, 2*i)   = mu(:, i) + eps(:, i);
%     end
%     
%     gmm.w  = new_w;
%     gmm.mu = new_mu;
%     gmm.sigma = new_sigma;
% end

function gmm = gmm_mixup(gmm)
    % perform a binary split of the GMM hyperparameters
    mu = gmm.mu; 
    sigma = gmm.sigma; 
    w = gmm.w;
    
    [ndim, nmix] = size(sigma);
    [sig_max, arg_max] = max(sigma);
    eps = sparse(0 * mu);
    eps(sub2ind([ndim, nmix], arg_max, 1 : nmix)) = sqrt(sig_max);
    % only perturb means associated with the max std along each dim 
    mu = [mu - eps, mu + eps];
    mu = mu;
    % mu = [mu - 0.2 * eps, mu + 0.2 * eps]; % HTK style
    sigma = [sigma, sigma];
    w = [w, w] * 0.5;
    
    gmm.w  = w;
    gmm.mu = mu;
    gmm.sigma = sigma;
end


function data = load_one_file(filename)
% filename = fullfile(feature_path, [file_list{i}, '.h5']);
    data = h5read(filename, '/data');
    datasize = size(data);
    if datasize(2) == 1
        data = data';
    end
    data = double(data');
end

