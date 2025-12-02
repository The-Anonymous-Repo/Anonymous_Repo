function  [CFA, P] = PCABM3DsubstractCDnoise_monochrome(nCFA,Rsigma,Gsigma,Bsigma,pattern)

    if strcmp(pattern,'2')
      sigma1=Gsigma;
      sigma2=Rsigma;
      sigma3=Bsigma;
      sigma4=Gsigma;
    elseif strcmp(pattern,'3')
      sigma1=Rsigma;
      sigma2=Gsigma;
      sigma3=Gsigma;
      sigma4=Bsigma;
    elseif strcmp(pattern,'4')
      sigma1=Bsigma;
      sigma2=Gsigma;
      sigma3=Gsigma;
      sigma4=Rsigma;
    elseif strcmp(pattern,'1')
      sigma1=Gsigma;
      sigma2=Bsigma;
      sigma3=Rsigma;
      sigma4=Gsigma;
    end
          
            Z1=nCFA(1:2:size(nCFA,1), 1:2:size(nCFA,2));
            Z2=nCFA(1:2:size(nCFA,1), 2:2:size(nCFA,2));
            Z3=nCFA(2:2:size(nCFA,1), 1:2:size(nCFA,2));           
            Z4=nCFA(2:2:size(nCFA,1), 2:2:size(nCFA,2));

 
            
            Z(:,:,1)=Z1;
            Z(:,:,2)=Z2;
            Z(:,:,3)=Z3;
            Z(:,:,4)=Z4;
            
            
            %getCov
            X(:,1)=Z1(:)-(sum(Z1(:))/size(Z1(:),1));
            X(:,2)=Z2(:)-(sum(Z2(:))/size(Z2(:),1));
            X(:,3)=Z3(:)-(sum(Z3(:))/size(Z3(:),1));
            X(:,4)=Z4(:)-(sum(Z4(:))/size(Z4(:),1));
            CovX=(X'*X)/(size(Z1(:),1)-1);
            
            CovX=CovX - diag([sigma1^2 sigma2^2 sigma3^2 sigma4^2]);
            
            [P,V]=eig(CovX);

            
            %Y=P'*X
            out=Matcolorconvert(Z,inv(P),0);

           %confirming Cov is diag
            out1=out(:,:,1);
            out2=out(:,:,2);
            out3=out(:,:,3);
            out4=out(:,:,4);
            Y(:,1)=out1(:)-(sum(out1(:))/size(out1(:),1));
            Y(:,2)=out2(:)-(sum(out2(:))/size(out2(:),1));
            Y(:,3)=out3(:)-(sum(out3(:))/size(out3(:),1));
            Y(:,4)=out4(:)-(sum(out4(:))/size(out4(:),1));
            CovY=(Y'*Y)/(size(Z1(:),1)-1);
            
            
            T=inv(P);
            max=[0 0 0 0];
            min=[0 0 0 0];
            for i=1:4
                for j=1:4
                    if T(i,j)>0
                        max(i)=max(i)+T(i,j);
                    else
                        min(i)=min(i)+T(i,j);
                    end
                end
            end
            max=max*255.0;
            min=min*255.0;
            
           
            %[min max] to [0 255]
            for i=1:4
             out0to255(:,:,i)=(255.0/(max(i)-min(i)))*out(:,:,i)-(255.0*min(i)/(max(i)-min(i)));
             BM3Dsigma(i)=sqrt( (T(i,1)*sigma1)^2 + (T(i,2)*sigma2)^2 +(T(i,3)*sigma3)^2 +(T(i,4)*sigma4)^2 );
             
             BM3Dsigma(i)=(255.0/(max(i)-min(i))) * BM3Dsigma(i);
            end
            
            BM3Dout0to1 = wholeBM3D(out0to255,BM3Dsigma);
  
            %[0 1] to [min max]
            for i=1:4
             BM3Dout(:,:,i)=(max(i)-min(i))*BM3Dout0to1(:,:,i)+min(i);
            end
           
            
            %INV-PCA 
            
            denoisedGRBG=Matcolorconvert(BM3Dout,P,0);
           
            CFA(1:2:size(nCFA,1), 1:2:size(nCFA,2)) = denoisedGRBG(:,:,1);
            CFA(1:2:size(nCFA,1), 2:2:size(nCFA,2)) = denoisedGRBG(:,:,2);
            CFA(2:2:size(nCFA,1), 1:2:size(nCFA,2)) = denoisedGRBG(:,:,3);
            CFA(2:2:size(nCFA,1), 2:2:size(nCFA,2)) = denoisedGRBG(:,:,4);
            
        
   
end

