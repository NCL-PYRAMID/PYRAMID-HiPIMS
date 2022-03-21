clear,clc
infilename = 'Landuse.tif';
[Landuse,R] = readgeoraster(infilename);
info = geotiffinfo(infilename);

[m,n]=size(Landuse);

geoTags = info.GeoTIFFTags.GeoKeyDirectoryTag;
tiffTags = struct('TileLength',1024,'TileWidth',1024);

rainfall = Landuse;
for i = 1:m
    for j = 1:n
        rainfall(i,j) = 0; 
    end
end
geotiffwrite('rainfallMask.tif',rainfall,R,'TiffType','bigtiff', ...
                             'GeoKeyDirectoryTag',geoTags, ...
                             'TiffTags',tiffTags)