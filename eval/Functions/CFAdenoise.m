function dnRaw=CFAdenoise(nRaw, Nsigma, pattern)

% Change range to [0 255]
nRaw = nRaw*255;

patterns = cell(2,2);
if( strcmpi( pattern, 'grbg' ) )
 patterns{1}{1} = 'grbg'; 
 patterns{1}{2} = 'rggb'; 
 patterns{2}{1} = 'bggr'; 
 patterns{2}{2} = 'gbrg'; 
 
elseif( strcmpi( pattern, 'gbrg' ) )
 patterns{1}{1} = 'gbrg'; 
 patterns{1}{2} = 'bggr'; 
 patterns{2}{1} = 'rggb'; 
 patterns{2}{2} = 'grbg'; 

elseif( strcmpi( pattern, 'rggb' ) )
 patterns{1}{1} = 'rggb'; 
 patterns{1}{2} = 'grbg'; 
 patterns{2}{1} = 'gbrg'; 
 patterns{2}{2} = 'bggr'; 

elseif( strcmpi( pattern, 'bggr' ) )
 patterns{1}{1} = 'bggr'; 
 patterns{1}{2} = 'gbrg'; 
 patterns{2}{1} = 'grbg'; 
 patterns{2}{2} = 'rggb'; 

else
 dnRaw = [];
 return
end

s = size(nRaw);
val = zeros(s);
num = zeros(s);

for row=1:2
 for col=1:2
  nr = nRaw(row:s(1), col:s(2));
  nr = nr(1:floor(size(nr,1)/2)*2, 1:floor(size(nr,2)/2)*2);
  dn = PCABM3DsubtractCDnoise(nr, Nsigma(1), Nsigma(2), Nsigma(3), patterns{row}{col} );
  
  s1 = row+size(dn,1)-1;
  s2 = col+size(dn,2)-1;
  val(row:s1, col:s2) = val(row:s1, col:s2) + dn;
  num(row:s1, col:s2) = num(row:s1, col:s2) + ones(size(dn));
 end
end

dnRaw = val ./ num;

% Change range back to [0 1]
dnRaw = dnRaw / 255;

end

