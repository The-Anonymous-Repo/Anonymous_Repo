import torch
import scipy.io as sio

# load as pytorch tensor from tokyo tech data mat file http://www.ok.sc.e.titech.ac.jp/res/PolarDem/index.html

def load_mat_as_tensor( mat_file ):
    data = sio.loadmat( mat_file )

    gt000 = torch.from_numpy( data["GT_0"] )
    gt045 = torch.from_numpy( data["GT_45"] )
    gt090 = torch.from_numpy( data["GT_90"] )
    gt135 = torch.from_numpy( data["GT_135"] )

    tt000 = torch.from_numpy( data["Test_0"] )
    tt045 = torch.from_numpy( data["Test_45"] )
    tt090 = torch.from_numpy( data["Test_90"] )
    tt135 = torch.from_numpy( data["Test_135"] )

    clean = torch.stack( [gt000, gt045, gt090, gt135], dim=0 )
    noisy = torch.stack( [gt000, gt045, gt090, gt135], dim=0 )

    clean = clean.permute( 0, 3, 1, 2 )
    noisy = noisy.permute( 0, 3, 1, 2 )

    return noisy, clean

if __name__ == "__main__" :
    noisy, clean = load_mat_as_tensor( "Scene1.mat" )
    print( noisy.shape, clen.shap )
