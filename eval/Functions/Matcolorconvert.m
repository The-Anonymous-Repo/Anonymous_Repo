function out = Matcolorconvert(rgb,A,B)
  rgb=double(rgb);
  A=double(A);
  B=double(B);
       %y=Ax+B
       out =zeros(size(rgb,1),size(rgb,2),size(A,1));
       
       for i=1:size(A,1)
           for j=1:size(A,2)
               out(:,:,i)=out(:,:,i)+A(i,j)*rgb(:,:,j);
           end
           
           if B~=0
           out(:,:,i)=out(:,:,i)+B(i);
           end
       end
end