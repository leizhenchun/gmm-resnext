function voxceleb_train_ubm()

nmix = 1024;
voxceleb_path = '/home/lzc/lzc/voxceleb';
exp_path      = '/home/lzc/lzc/voxceleb_exp/GMM_MFCC80/';

make_path(exp_path);

diary(fullfile(exp_path, ['VOXCELEB_GMM_MFCC80_', num2str(nmix), '_log.txt']));
diary on;

voxceleb_gmm_run(voxceleb_path, exp_path, nmix);

voxceleb_gmm_llk_mean_std_run(voxceleb_path, exp_path, nmix);

diary off;

end

function voxceleb_gmm_run(voxceleb_path, exp_path, nmix)


train_feat_path = fullfile(voxceleb_path, 'voxceleb2_mfcc80');
train_list_file = fullfile(voxceleb_path, 'train_list.txt');
meta_file       = fullfile(voxceleb_path, 'vox2_meta.csv');

gmmModelFile    = fullfile(exp_path, ['VOXCELEB2_GMM_MFCC80_', num2str(nmix)]);


%% GMM training
show_message('Load training data ...');

[ speakerId, train_feat_list ] = voxceleb_read_train_list( train_list_file );
% selected = randperm(length(train_feat_list), 6000);
% train_feat_list = train_feat_list(selected);
show_message(['Training dataset size : ', num2str(length(train_feat_list))]);

% train_feature = voxceleb_load_feature( train_feat_path, train_feat_list );

show_message('Training UBM ...');
ubm_all = voxceleb_gmm_em_file(train_feat_path, train_feat_list, nmix);


save(gmmModelFile, 'ubm_all');
clear train_feature;
show_message('Done!');

end



function voxceleb_gmm_llk_mean_std_run(voxceleb_path, exp_path, nmix)
show_message(['voxceleb_gmm_llk_mean_std ']);

train_feat_path = fullfile(voxceleb_path, 'voxceleb2_mfcc80');
train_list_file = fullfile(voxceleb_path, 'train_list.txt');
meta_file       = fullfile(voxceleb_path, 'vox2_meta.csv');

gmmModelFile    = fullfile(exp_path, ['VOXCELEB2_GMM_MFCC80_', num2str(nmix)]);

[ speakerId, train_feat_list ] = voxceleb_read_train_list( train_list_file );

% train_feat_list = train_feat_list(1: 10000);

show_message(['loading GMM : ', gmmModelFile]);
load(gmmModelFile, 'ubm_all');


for gmm_idx = 1 : length(ubm_all)
    
    ubm = ubm_all{gmm_idx};
%     gmm_bonafide = gmm_bonafide_all{gmm_idx};
%     gmm_spoof = gmm_spoof_all{gmm_idx};
    
    nmix = length(ubm.w);
    
    
    show_message(['compute GMM llk : nmix ', num2str(nmix)]);
    
    [ubm_mean, ubm_std] = compute_mean_std(ubm, train_feat_path, train_feat_list);
    
    gmm_filename = fullfile(exp_path, ['VOXCELEB_GMM_MFCC80_', num2str(nmix), '.h5']);
    save_gmm(gmm_filename, ubm.w, ubm.mu, ubm.sigma, ubm_mean, ubm_std);

% %     show_message( 'compute GMM llk : bonafide');
%     [bonafide_mean, bonaffide_std] = compute_mean_std(gmm_bonafide, train_feature);
%     gmm_filename = fullfile(exp_path, ['ASVspoof2019_GMM_', feat_acc, '_', num2str(nmix), '_bonafide.h5']);
%     save_gmm(gmm_filename, gmm_bonafide.w, gmm_bonafide.mu, gmm_bonafide.sigma, bonafide_mean, bonaffide_std);
% 
% %     show_message( 'compute GMM llk : spoof');
%     [spoof_mean, spoof_std] = compute_mean_std(gmm_spoof, train_feature);
%     gmm_filename = fullfile(exp_path, ['ASVspoof2019_GMM_', feat_acc, '_', num2str(nmix), '_spoof.h5']);
%     save_gmm(gmm_filename, gmm_spoof.w, gmm_spoof.mu, gmm_spoof.sigma, spoof_mean, spoof_std);

end


end


function logprob = lgmmprob2(data, mu, sigma, w)
    logprob = (1./sigma)' * (data .* data) - 2 * (mu./sigma)' * data;
end

function [feat_mean, feat_std] = compute_mean_std(gmm, feature_path, feature_list)

    feat_x1 = zeros(1, size(gmm.mu, 2));
    feat_xx1 = zeros(1, size(gmm.mu, 2));
    feat_count = 0;
    
    fileCount = length(feature_list);
    parfor i=1:fileCount

        feature_file = fullfile(feature_path, [feature_list{i}, '.h5']);
        feature = h5read(feature_file, '/data');
        datasize = size(feature);
        if datasize(2) == 1
            feature = feature';
        end
        feature = double(feature');

        feat_count = feat_count + size(feature, 2);

        post1 = lgmmprob2(feature, gmm.mu, gmm.sigma, gmm.w');
        feat_x1 = feat_x1 + sum(post1, 2)';
        feat_xx1 = feat_xx1 + sum(post1 .* post1, 2)';
        
            
        if mod(i, 100000) == 0
            show_message([num2str(i), '/', num2str(fileCount)]);
        end
        
    end
    
    feat_mean = feat_x1 / feat_count;
    feat_std = sqrt(feat_xx1 / feat_count - feat_mean .* feat_mean);
    
end



function [gm, gv] = comp_gm_gv(data)
    % computes the global mean and variance of data
    nframes = cellfun(@(x) size(x, 2), data, 'UniformOutput', false);
    nframes = sum(cell2mat(nframes));
    gm = cellfun(@(x) sum(x, 2), data, 'UniformOutput', false);
    gm = sum(cell2mat(gm'), 2)/nframes;
    gv = cellfun(@(x) sum(bsxfun(@minus, x, gm).^2, 2), data, 'UniformOutput', false);
    gv = sum(cell2mat(gv'), 2)/( nframes - 1 );
end


function save_gmm(gmm_filename, w, mu, sigma, feat_mean, feat_std)

h5create(gmm_filename, '/w', size(w));
h5write( gmm_filename, '/w', w);

h5create(gmm_filename, '/mu', size(mu));
h5write( gmm_filename, '/mu', mu);

h5create(gmm_filename, '/sigma', size(sigma));
h5write( gmm_filename, '/sigma', sigma);

h5create(gmm_filename, '/feat_mean', size(feat_mean));
h5write( gmm_filename, '/feat_mean', feat_mean);

h5create(gmm_filename, '/feat_std', size(feat_std));
h5write( gmm_filename, '/feat_std', feat_std);


end


