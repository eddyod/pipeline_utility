function transform(img_path, vtk_path, output_dir)
    % LASTN = maxNumCompThreads(16)

    parpool('local',1);
    pctRunOnAll maxNumCompThreads(1);
    imglist = dir(strcat(img_path, '/', '*.jp2'));
    % vtklist = dir(strcat(vtk_path, 'input_to_registered_displacement*.vtk'));
    % vtklist = dir(strcat(vtk_path, 'registered_to_input_displacement*.vtk'));
    vtklist = dir(strcat(vtk_path, '/', '*.vtk'));

    [count, a] = size(imglist);
    parfor i = 1 : count
        % LASTN = maxNumCompThreads(1)
        maxNumCompThreads()
        imgpath = (strcat(imglist(i).folder, '/', imglist(i).name));
        vtkpath = (strcat(vtklist(i).folder, '/', vtklist(i).name));
        disp(imgpath)
        disp(vtkpath)
        Jhigh = imread(imgpath);
        Jhigh = im2double(Jhigh);


        % !FIXME due to space and memory limit, use lossy resolution(dowmsaple 2 times)
        Jhigh = imresize(Jhigh,0.5,'nearest');
        % get its pixel size in um
        dxhigh = [0.325 * 2, 0.325 * 2]; % just guessing here, use your actual pixel size


        % get the number of pixels
        nxhigh = [size(Jhigh,2),size(Jhigh,1)];
        % get the domain
        xhigh = (0:nxhigh(1)-1)*dxhigh(1);
        yhigh = (0:nxhigh(2)-1)*dxhigh(2);
        % make sure it is centered at 0
        xhigh = xhigh - mean(xhigh);
        yhigh = yhigh - mean(yhigh);
        % use meshgrid for sampling
        [XHIGH,YHIGH] = meshgrid(xhigh,yhigh);


        % load the transformation
        % [x, y, z, AJphiJAxyX, AJphiJAxyY] = transform_aux(vtkpath);
        disp(strcat('entering read_vtk_image with filename: ', vtkpath))
        [x,y,z,I,title,names] = read_vtk_image(vtkpath);
        [X,Y] = meshgrid(x,y);

        AJphiJAxyX = I(:,:,1,1)+X;
        AJphiJAxyY = I(:,:,1,2)+Y;
        % upsample the transformation
        F = griddedInterpolant({y,x},AJphiJAxyX,'linear','nearest'); % note x,y are
        %loaded from the mat file, note always use order y,x
        AJphiJXhigh = F(YHIGH,XHIGH);
        F = griddedInterpolant({y,x},AJphiJAxyY,'linear','nearest');
        AJphiJYhigh = F(YHIGH,XHIGH);

        % apply transformation to your image
        transformed_Jhigh = zeros(size(Jhigh));
        for c = 1 : size(Jhigh,3)
            F = griddedInterpolant({yhigh,xhigh},Jhigh(:,:,c),'linear','none');
            transformed_Jhigh(:,:,c) = F(AJphiJYhigh,AJphiJXhigh);
        end

        transformed_Jhigh(isnan(transformed_Jhigh)) = 255;
        transformed_Jhigh=im2uint8(transformed_Jhigh);
        fileout = strcat(output_dir, '/', imglist(i).name(1:end),'.tif')
        disp(strcat('writing ', fileout))
        imwrite(transformed_Jhigh, fileout)


    end
end


% for i = 1:50
%     %try
%         vtkpath = (strcat(reconlist(i).folder, '/', reconlist(i).name));
%         imgpath = (strcat(imglist(i).folder, '/', imglist(i).name));
%         disp(imgpath)
%         %if exist(strcat('/scratch/783UPDATE/reg_high_tif/', imglist(i).name(1:end-13),'.tif'),'file')==2
%            % disp('we got it!')
%             %continue
%         %end
%         % load your high res image for the "f"th slice
%         Jhigh = imread(imgpath);
%         Jhigh = im2double(Jhigh);

%         Jhigh = imresize(Jhigh,0.5,'nearest');
%         % get its pixel size in um
%         dxhigh = [0.46 * 2, 0.46 * 2]; % just guessing here, use your actual pixel size
%         % get the number of pixels
%         nxhigh = [size(Jhigh,2),size(Jhigh,1)];
%         % get the domain
%         xhigh = (0:nxhigh(1)-1)*dxhigh(1);
%         yhigh = (0:nxhigh(2)-1)*dxhigh(2);
%         % make sure it is centered at 0
%         xhigh = xhigh - mean(xhigh);
%         yhigh = yhigh - mean(yhigh);
%         % use meshgrid for sampling
%         [XHIGH,YHIGH] = meshgrid(xhigh,yhigh);

%         % load the transformation
%         load(vtkpath)
%         % upsample the transformation
%         F = griddedInterpolant({y,x},AJphiJAxyX,'linear','nearest'); % note x,y are
%         %loaded from the mat file, note always use order y,x
%         AJphiJXhigh = F(YHIGH,XHIGH);
%         F = griddedInterpolant({y,x},AJphiJAxyY,'linear','nearest');
%         AJphiJYhigh = F(YHIGH,XHIGH);

%         % apply transformation to your image
%         transformed_Jhigh = zeros(size(Jhigh));
%             for c = 1 : size(Jhigh,3)
%                 F = griddedInterpolant({yhigh,xhigh},Jhigh(:,:,c),'linear','none');
%                 transformed_Jhigh(:,:,c) = F(AJphiJYhigh,AJphiJXhigh);
%             end

%         transformed_Jhigh(isnan(transformed_Jhigh)) = 255;
%         transformed_Jhigh=im2uint8(transformed_Jhigh);
%         imwrite(transformed_Jhigh, strcat('/scratch/783UPDATE/reg_high_tif/', imglist(i).name(1:end-13),'.tif'))
%     %catch
%         %disp(strcat(imgpath,' not working!'))
%     %end
% end

