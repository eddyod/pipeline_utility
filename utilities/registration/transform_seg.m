function transform_seg(seg_file, input_dir, animal, res)

% seg file is the filename of the atlas you want to deform (vtk)
% input_dir is a directory of tif images you want to match, with a
% geometry csv file in the same directory
% output_dir is the directory that contains the outputs from registration
% res is the desired resolution for the data (e.g. microns per pixel)
%
% this function has no outputs, it just makes a picture.  You can save
% outputs by editing the bottom

%

addpath Functions/vtk

addpath Functions/plotting
% if nargin == 0
%     % example inputs
%     seg_file = 'annotation_50.vtk';
%     input_dir = 'MD720/';
%     output_dir = 'MD720_OUT/';
%     res = 5; % e.g. 5 um per pixel
% end


%%
% load the atlas
[xI,yI,zI,I,title_,names] = read_vtk_image(seg_file);
rng(1);
colors = rand(256,3);
colors(1,:) = 1; % set background white

%%
% get geometry of files from input dir
geometry_file = strcat('/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/', animal, '/brains_info/geometry.csv');
% fid = fopen([input_dir geometry_file(1).name],'rt');
fid = fopen(geometry_file,'rt');
line = fgetl(fid); % ignore the first line
% it should say
% filename, nx, ny, nz, dx, dy, dz, x0, y0, z0

csv_data = {};
count = 0;
while 1
    line = fgetl(fid);
    if line == -1
        break
    end
    count = count + 1;
    % process this line, splitting at commas
    csv_data(count,:) = strsplit(line,',');
    %
end
fclose(fid);
files = csv_data(:,1);
nxJ = cellfun(@(x)str2num(x), csv_data(:,2:3));
dxJ = cellfun(@(x)str2num(x), csv_data(:,5:6));

x0J = cellfun(@(x)str2num(x), csv_data(:,8:9));
z0J = cellfun(@(x)str2num(x), csv_data(:,10));

zJ = z0J;
dzJ = cellfun(@(x) str2num(x), csv_data(:,7));

for f = 1 : length(files)
    xJ{f} = x0J(f,1) + (0:nxJ(f,1)-1)*dxJ(f,1);
    yJ{f} = x0J(f,2) + (0:nxJ(f,2)-1)*dxJ(f,2);
end


%%
% loop through the files
% for each file apply one transform to the atlas, and one to the target, so
% they meet in registered space

for f = 1 : length(files)
    [directory,fname,ext] = fileparts(files{f});

    % load the tif image and display
    tifpath = strcat(input_dir, '/', fname, '.tif');
    disp(tifpath);
    % J = imread([input_dir files{f}]);
    J = imread(tifpath);
    if strcmp(class(J),'uint8')
        J = double(J)/255.0;
    end
    J = double(J);

    % transform this image to registered space (2d transforms, identity in z)
    % this involves two steps
    % 1. upsample the transform to resolution res
    % 2. resample the data
    %
    % load the transform
    try
        % [xTJ,yTJ,zTJ,DeltaTJ,title_,names] = read_vtk_image([output_dir 'registered_to_input_displacement_' fname '.vtk']);
        vtkfile = strcat('/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/', animal, '/preps/vtk/input/', fname, '.vtk');
        [xTJ,yTJ,zTJ,DeltaTJ,title_,names] = read_vtk_image(vtkfile);
    catch
        disp(['Could not read ' fname]);
        continue
    end

    % upsample the transform
    xTJup = xTJ(1) : res : xTJ(end);
    yTJup = yTJ(1) : res : yTJ(end);
    [XTJup,YTJup] = meshgrid(xTJup,yTJup);
    DeltaTJup = zeros(size(XTJup,1),size(XTJup,2),1,2);
    F = griddedInterpolant({yTJ,xTJ},DeltaTJ(:,:,1,1),'linear','nearest');
    DeltaTJup(:,:,1,1) = F(YTJup,XTJup);
    F = griddedInterpolant({yTJ,xTJ},DeltaTJ(:,:,1,2),'linear','nearest');
    DeltaTJup(:,:,1,2) = F(YTJup,XTJup);

    phiTJup = zeros(size(DeltaTJup));
    phiTJup(:,:,:,1) = DeltaTJup(:,:,:,1) + XTJup;
    phiTJup(:,:,:,2) = DeltaTJup(:,:,:,2) + YTJup;

    % apply transform to data
    JT = zeros(size(phiTJup,1),size(phiTJup,2),size(J,3));
    for c = 1 : size(J,3)
        F = griddedInterpolant({yJ{f},xJ{f}},J(:,:,c),'linear','nearest');
        JT(:,:,c) = F(phiTJup(:,:,:,2),phiTJup(:,:,:,1));
    end

    % transform the atlas
    % load the transform
    fileinput = strcat('/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/', animal, '/preps/vtk/atlas/', fname, '.vtk')
    [xTI,yTI,zTI,DeltaTI,title_,names] = read_vtk_image(fileinput);

    % upsample it
    DeltaTIup = zeros(size(XTJup,1),size(XTJup,2),1,3);

    F = griddedInterpolant({yTJ,xTJ},DeltaTI(:,:,1,1),'linear','nearest');
    DeltaTIup(:,:,1,1) = F(YTJup,XTJup);
    F = griddedInterpolant({yTJ,xTJ},DeltaTI(:,:,1,2),'linear','nearest');
    DeltaTIup(:,:,1,2) = F(YTJup,XTJup);
    F = griddedInterpolant({yTJ,xTJ},DeltaTI(:,:,1,3),'linear','nearest');
    DeltaTIup(:,:,1,3) = F(YTJup,XTJup);

    phiTIup = zeros(size(DeltaTJup));
    phiTIup(:,:,:,1) = DeltaTIup(:,:,:,1) + XTJup;
    phiTIup(:,:,:,2) = DeltaTIup(:,:,:,2) + YTJup;
    phiTIup(:,:,:,3) = DeltaTIup(:,:,:,3) + zJ(f);

    % apply transform to data
    [XTI,YTI,ZTI] = meshgrid(xTI,yTI,zTI);

    F = griddedInterpolant({yI,xI,zI},I,'nearest','nearest'); % for annotation use nearest
    IT = F(phiTIup(:,:,:,2),phiTIup(:,:,:,1),phiTIup(:,:,:,3));
    outpath = strcat('/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/', animal, '/preps/vtk/mat/', fname, '.mat')
    save(outpath, 'IT')

end
