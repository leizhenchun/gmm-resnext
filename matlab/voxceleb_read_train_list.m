function [ speakerId, waveFile ] = voxceleb_read_train_list( filename )

fileID = fopen(filename);
protocol = textscan(fileID, '%s%s');
fclose(fileID);

        
speakerId  = protocol{1};
waveFile   = protocol{2};



for i = 1 : length(waveFile)
    waveFile{i} = waveFile{i}(1:length(waveFile{i}) - 4);
    %     fileId{i} = strrep(fileId{i}, '.wav', '');
end

end