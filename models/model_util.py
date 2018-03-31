import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm


def get_full_model(net, res, n_class, input_ch, is_data_parallel=True):
    if net == "fcn":
        from models.fcn import ResFCN
        model = ResFCN(n_class, res, input_ch)

    elif net == "fcnvgg":
        from models.vgg_fcn import FCN8s
        # TODO suport input_ch
        model = FCN8s(n_class)

    elif "drn" in net:
        from models.dilated_fcn import DRNSeg
        assert net in ["drn_c_26", "drn_c_42", "drn_c_58", "drn_d_22", "drn_d_38", "drn_d_54", "drn_d_105"]
        model = DRNSeg(net, n_class, input_ch=input_ch)

    elif net == "psp":
        from models.pspnet import PSPNet
        model = PSPNet(n_class, layer=res, input_ch=input_ch)

    elif net == "unet":
        from models.unet import UNet
        model = UNet(n_classes=n_class, input_ch=input_ch)

    elif net == "fusenet":
        from models.FuseNet import FuseNet
        model = FuseNet(n_class=n_class, input_ch=input_ch)

    else:
        raise NotImplementedError("Only FCN, SegNet, PSPNet, DRNet, UNet are supported!")

    if is_data_parallel:
        return torch.nn.DataParallel(model)

    return model


def get_full_uncertain_model(net, res, n_class, input_ch, criterion, is_data_parallel=True, n_dropout=10):
    if "drn" in net:
        from models.dilated_fcn import UncertainDRNSeg
        assert net in ["drn_c_26", "drn_c_42", "drn_c_58", "drn_d_22", "drn_d_38", "drn_d_54", "drn_d_105"]
        model = UncertainDRNSeg(net, n_class, input_ch=input_ch, n_dropout_exp=n_dropout, criterion=criterion)

    else:
        raise NotImplementedError("Only FCN, SegNet, PSPNet, DRNet, UNet are supported!")

    if is_data_parallel:
        return torch.nn.DataParallel(model)

    return model


def get_multichannel_model(net_name, input_ch_list, n_class, method="MCD", res="50", is_data_parallel=True):
    from models.dilated_fcn import FusionDRNSegPixelClassifier

    frond_model_list = []

    if "drn" in net_name:
        from models.dilated_fcn import DRNSegBase, DRNSegPixelClassifier
        fusion_type = method.split("-")[-1]
        ver = "ver2" if "ver2" in net_name else "ver1"
        drn_name = net_name.replace("_ver2", "")

        for input_ch in input_ch_list:
            frond_model_list.append(DRNSegBase(model_name=drn_name, n_class=n_class, input_ch=input_ch, ver=ver))

        end_model = FusionDRNSegPixelClassifier(fusion_type=fusion_type, n_class=n_class, ver=ver)
    else:
        raise NotImplementedError("Only DRN are supported!")

    if is_data_parallel:
        return [torch.nn.DataParallel(x) for x in frond_model_list], torch.nn.DataParallel(end_model)

    return frond_model_list, end_model


def get_multitask_models(net_name, input_ch, n_class, semseg_criterion=None, discrepancy_criterion=None,
                         is_data_parallel=False, is_src_only=False):
    if "drn" in net_name:
        from models.dilated_fcn import MultiTaskEncoder, MultiTaskDecoder, MCDMultiTaskDecoder
        model_enc = MultiTaskEncoder(model_name=net_name, input_ch=3)  # RGB is 3 channel

        if is_src_only:
            model_dec = MultiTaskDecoder(n_class=n_class, depth_ch=input_ch - 3, semseg_criterion=semseg_criterion,
                                         discrepancy_criterion=discrepancy_criterion)
        else:
            model_dec = MCDMultiTaskDecoder(n_class=n_class, depth_ch=input_ch - 3, semseg_criterion=semseg_criterion,
                                            discrepancy_criterion=discrepancy_criterion)
    else:
        raise NotImplementedError("Only FCN (Including Dilated FCN), SegNet, PSPNet UNet are supported!")

    if is_data_parallel:
        return torch.nn.DataParallel(model_enc), torch.nn.DataParallel(model_dec)

    return model_enc, model_dec


def get_triple_multitask_models(net_name, input_ch, n_class, semseg_criterion=None, discrepancy_criterion=None,
                                is_data_parallel=False, semseg_shortcut=False, depth_shortcut=False,
                                add_pred_seg_boundary_loss=False, is_src_only=False, use_seg2bd_conv=False):
    # TODO input_ch is ignored...

    if "drn" in net_name:
        from models.dilated_fcn import MultiTaskEncoderReturningMultipleFeaturemaps, TripleMultiTaskDecoder, \
            MCDTripleMultiTaskDecoder
        model_enc = MultiTaskEncoderReturningMultipleFeaturemaps(model_name=net_name, input_ch=3)  # RGB is 3 channel

        if is_src_only:
            model_dec = TripleMultiTaskDecoder(n_class=n_class, depth_ch=3,
                                               semseg_criterion=semseg_criterion,
                                               semseg_shortcut=semseg_shortcut, depth_shortcut=depth_shortcut,
                                               add_pred_seg_boundary_loss=add_pred_seg_boundary_loss)
        else:
            model_dec = MCDTripleMultiTaskDecoder(n_class=n_class, depth_ch=3,
                                                  semseg_criterion=semseg_criterion,
                                                  discrepancy_criterion=discrepancy_criterion,
                                                  semseg_shortcut=semseg_shortcut, depth_shortcut=depth_shortcut,
                                                  add_pred_seg_boundary_loss=add_pred_seg_boundary_loss,
                                                  use_seg2bd_conv=use_seg2bd_conv)
    else:
        raise NotImplementedError("Only FCN (Including Dilated FCN), SegNet, PSPNet UNet are supported!")

    if is_data_parallel:
        return torch.nn.DataParallel(model_enc), torch.nn.DataParallel(model_dec)

    return model_enc, model_dec


def get_segbd_multitask_models(net_name, input_ch, n_class, semseg_criterion=None, discrepancy_criterion=None,
                               is_data_parallel=False, semseg_shortcut=False, depth_shortcut=False,
                               add_pred_seg_boundary_loss=False, is_src_only=False, use_seg2bd_conv=False):
    # TODO input_ch is ignored...

    if "drn" in net_name:
        from models.dilated_fcn import MultiTaskEncoderReturningMultipleFeaturemaps, MCDSegBDMultiTaskDecoder
        model_enc = MultiTaskEncoderReturningMultipleFeaturemaps(model_name=net_name, input_ch=3)  # RGB is 3 channel

        if is_src_only:
            raise NotImplementedError()
        else:
            model_dec = MCDSegBDMultiTaskDecoder(n_class=n_class, depth_ch=3,
                                                 semseg_criterion=semseg_criterion,
                                                 discrepancy_criterion=discrepancy_criterion,
                                                 semseg_shortcut=semseg_shortcut,
                                                 add_pred_seg_boundary_loss=add_pred_seg_boundary_loss,
                                                 use_seg2bd_conv=use_seg2bd_conv)
    else:
        raise NotImplementedError("Only FCN (Including Dilated FCN), SegNet, PSPNet UNet are supported!")

    if is_data_parallel:
        return torch.nn.DataParallel(model_enc), torch.nn.DataParallel(model_dec)

    return model_enc, model_dec


def get_models(net_name, input_ch, n_class, res="50", method="MCD", is_data_parallel=False):
    def get_MCD_model_list():
        if net_name == "fcn":
            from models.fcn import ResBase, ResClassifier
            model_g = ResBase(n_class, layer=res, input_ch=input_ch)
            model_f1 = ResClassifier(n_class)
            model_f2 = ResClassifier(n_class)
        elif net_name == "fcnvgg":
            from models.vgg_fcn import FCN8sBase, FCN8sClassifier
            # TODO implement input_ch
            model_g = FCN8sBase(n_class)
            model_f1 = FCN8sClassifier(n_class)
            model_f2 = FCN8sClassifier(n_class)
        elif net_name == "psp":
            # TODO add "input_ch" argument
            from models.pspnet import PSPBase, PSPClassifier
            model_g = PSPBase(layer=res, input_ch=input_ch)
            model_f1 = PSPClassifier(num_classes=n_class)
            model_f2 = PSPClassifier(num_classes=n_class)
        elif net_name == "segnet":
            # TODO add "input_ch" argument
            from models.segnet import SegNetBase, SegNetClassifier
            model_g = SegNetBase()
            model_f1 = SegNetClassifier(n_class)
            model_f2 = SegNetClassifier(n_class)


        elif "drn" in net_name:
            if "fusenet" in net_name:
                drn_name = net_name.replace("_fusenet", "")

                from models.dilated_fcn import FuseDRNSegBase, DRNSegPixelClassifier
                model_g = FuseDRNSegBase(model_name=drn_name, n_class=n_class,
                                         input_ch=input_ch)
                model_f1 = DRNSegPixelClassifier(n_class=n_class)
                model_f2 = DRNSegPixelClassifier(n_class=n_class)

            else:
                ver = "ver2" if "ver2" in net_name else "ver1"
                drn_name = net_name.replace("_ver2", "")

                from models.dilated_fcn import DRNSegBase, DRNSegPixelClassifier
                model_g = DRNSegBase(model_name=drn_name, n_class=n_class, input_ch=input_ch, ver=ver)
                model_f1 = DRNSegPixelClassifier(n_class=n_class, ver=ver)
                model_f2 = DRNSegPixelClassifier(n_class=n_class, ver=ver)
        elif net_name == "unet":
            # TODO add "input_ch" argument
            from models.unet import UNetBase, UNetClassifier
            model_g = UNetBase(input_ch=input_ch)
            model_f1 = UNetClassifier(n_classes=n_class)
            model_f2 = UNetClassifier(n_classes=n_class)

        elif net_name == "fusenet":
            from models.FuseNet import FuseBase, FuseClassifier
            model_g = FuseBase(input_ch=input_ch)
            model_f1 = FuseClassifier(n_class=n_class)
            model_f2 = FuseClassifier(n_class=n_class)



        else:
            raise NotImplementedError("Only FCN (Including Dilated FCN), SegNet, PSPNet UNet are supported!")

        return model_g, model_f1, model_f2

    def get_mfnet_model_list():
        from models.unet import MultiUNetClassifier

        assert input_ch in [4, 6]
        if "drn" in net_name:
            ver = "ver2" if "ver2" in net_name else "ver1"

            use_score_fusion = True if "score" in method.lower() else False

            drn_name = net_name.replace("_ver2", "")

            from models.dilated_fcn import DRNSegBase, DRNSegPixelClassifier, ScoreFusionDRNSegPixelClassifier
            fusion_type = method.split("-")[-1]

            print ("fusion type: %s" % fusion_type)

            model_g_3ch = DRNSegBase(model_name=drn_name, n_class=n_class, input_ch=3, ver=ver)
            model_g_1ch = DRNSegBase(model_name=drn_name, n_class=n_class, input_ch=input_ch - 3, ver=ver)

            if use_score_fusion:
                print ("Score Fusion!!!")
                model_f1 = ScoreFusionDRNSegPixelClassifier(fusion_type=fusion_type, n_class=n_class)
                model_f2 = ScoreFusionDRNSegPixelClassifier(fusion_type=fusion_type, n_class=n_class)
            else:
                from models.dilated_fcn import FusionDRNSegPixelClassifier
                model_f1 = FusionDRNSegPixelClassifier(fusion_type=fusion_type, n_class=n_class, ver=ver)
                model_f2 = FusionDRNSegPixelClassifier(fusion_type=fusion_type, n_class=n_class, ver=ver)


        elif net_name == "unet":
            # TODO add "input_ch" argument
            from models.unet import UNetBase, UNetClassifier
            model_g_3ch = UNetBase(input_ch=3)
            model_g_1ch = UNetBase(input_ch=input_ch - 3)
            model_f1 = MultiUNetClassifier(n_classes=n_class)
            model_f2 = MultiUNetClassifier(n_classes=n_class)

        elif net_name == "fcn":
            # TODO add "input_ch" argument
            from models.fcn import ResBase, MFResClassifier2
            model_g_3ch = ResBase(num_classes=n_class, layer=res, input_ch=3)
            model_g_1ch = ResBase(num_classes=n_class, layer=res, input_ch=input_ch - 3)
            model_f1 = MFResClassifier2(n_class=n_class)
            model_f2 = MFResClassifier2(n_class=n_class)

        else:
            raise NotImplementedError("Only Dilated FCN is supported!")

        return model_g_3ch, model_g_1ch, model_f1, model_f2

    if method == "MCD":
        model_list = get_MCD_model_list()
    elif "MFNet" in method:
        model_list = get_mfnet_model_list()

    else:
        return NotImplementedError("Sorry... Only MCD is supported!")

    if is_data_parallel:
        return [torch.nn.DataParallel(x) for x in model_list]
    else:
        return model_list


def get_optimizer(model_parameters, opt, lr, momentum, weight_decay):
    if opt == "sgd":
        return torch.optim.SGD(filter(lambda p: p.requires_grad, model_parameters), lr=lr, momentum=momentum,
                               weight_decay=weight_decay)

    elif opt == "adadelta":
        return torch.optim.Adadelta(filter(lambda p: p.requires_grad, model_parameters), lr=lr,
                                    weight_decay=weight_decay)

    elif opt == "adam":
        return torch.optim.Adam(filter(lambda p: p.requires_grad, model_parameters), lr=lr, betas=[0.5, 0.999],
                                weight_decay=weight_decay)
    else:
        raise NotImplementedError("Only (Momentum) SGD, Adadelta, Adam are supported!")


def fix_batchnorm_when_training(model):
    if issubclass(type(model), _BatchNorm):
        model.training = False

    for module in model.children():
        fix_batchnorm_when_training(module)


def fix_dropout_when_training(model):
    if type(model) in [nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout]:
        model.training = False
        print ("Fixed one dropout layer")

    for module in model.children():
        fix_dropout_when_training(module)


def check_training(model):
    print (type(model))
    print (model.training)
    for module in model.children():
        check_training(module)
