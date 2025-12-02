function sample_download(path_to_dataset)

p = mfilename('fullpath');
[installDirec,] = fileparts(p);
if nargin ==0
    path_to_dataset = installDirec;
end
if ~exist(path_to_dataset)
    mkdir(path_to_dataset)
end

% download sample data
urlSampleData = 'http://www.ok.sc.e.titech.ac.jp/res/PolarDem/data/InputPDD.zip';
filenameSampleData= path_to_dataset;
unzip(urlSampleData,filenameSampleData)

% download BM3D code
urlSampleData = 'https://webpages.tuni.fi/foi/GCF-BM3D/BM3D.zip';
filenameSampleData= "BM3D";
unzip(urlSampleData,filenameSampleData)

end

