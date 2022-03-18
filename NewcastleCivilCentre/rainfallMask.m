[m,n]=size(Landuse);
rainfall = Landuse;
for i = 1:m
    for j = 1:n
        if Landuse(i,j) ~= -9999
            rainfall(i,j) = 0;
        end     
    end
end
imwrite(rainfall,'rainfallMask.tif')