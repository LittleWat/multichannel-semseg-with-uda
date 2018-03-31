function apply_bwboundary(indir)

  fn_list = dir(strcat(indir,'/*.png'));

  outdir = fullfile(fileparts(indir), 'bwboundary');
  mkdir(outdir);

  for i=1:length(fn_list)
    infn = fullfile(indir, fn_list(i).name);
    BW = imread(infn);
    [B,L,N,A] = bwboundaries(BW);
    L = L(2:end-1, 2:end-1);

    outfn =  fullfile(outdir,fn_list(i).name);
    imwrite(L/255, outfn)
  end

end

