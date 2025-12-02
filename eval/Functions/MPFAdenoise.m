function [dnRaw, P]=MPFAdenoise(nRaw, Nsigma, pattern)

% Change range to [0 255]
nRaw = nRaw * 255;

patterns = cell(2,2);
if( strcmpi( pattern, '1' ) ) % [90 45;135 0]
 patterns{1}{1} = '1'; 
 patterns{1}{2} = '3'; 
 patterns{2}{1} = '4'; 
 patterns{2}{2} = '2'; 
 
elseif( strcmpi( pattern, '2' ) ) % [45 90;0 135] 
 patterns{1}{1} = '2';  
 patterns{1}{2} = '4';  
 patterns{2}{1} = '3';  
 patterns{2}{2} = '1'; 

elseif( strcmpi( pattern, '3' ) ) % [135 0;90 45]
 patterns{1}{1} = '3';  
 patterns{1}{2} = '1'; 
 patterns{2}{1} = '2'; 
 patterns{2}{2} = '4';  

elseif( strcmpi( pattern, '4' ) ) % [0 135;45 90]
 patterns{1}{1} = '4'; 
 patterns{1}{2} = '2';  
 patterns{2}{1} = '1'; 
 patterns{2}{2} = '3';  

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
  [dn, P] = PCABM3DsubstractCDnoise_monochrome(nr, Nsigma, Nsigma, Nsigma, patterns{row}{col} );
  
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
