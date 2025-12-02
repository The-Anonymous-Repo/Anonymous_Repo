function rgbout = wholeBM3D(nrgb,BM3Dsigma)
    channel = size(nrgb,3);
    for i=1:size(nrgb,3)
        [PSNRR rgbout(:,:,i)]=BM3D(1,nrgb(:,:,i),BM3Dsigma(i),'np',0);
    end
end
