import os

import torch
import torchvision
from torch.autograd import Variable


class Solver(object):
    def __init__(self, args, data_loader):
        self.generator = None
        self.discriminator = None
        self.g_optimizer = None
        self.d_optimizer = None
        self.build_model()
        self.config = args

    def build_model(self):
        """Build generator and discriminator."""

        from models.dilated_fcn import DRNSegBase, DRNSegPixelClassifier, DRNSegDomainClassifier

        self.model_g = DRNSegBase(model_name=self.config.net, n_class=self.config.n_class)
        self.model_f = DRNSegPixelClassifier(n_class=self.config.n_class)
        self.model_d = DRNSegDomainClassifier(n_class=self.config.n_class)

        self.optimizer_g = torch.optim.SGD(self.model_g.parameters(), lr=self.config.lr, momentum=self.config.momentum,
                                           weight_decay=self.config.weight_decay)
        self.optimizer_d = torch.optim.SGD(self.model_d.parameters(), lr=self.config.lr, momentum=self.config.momentum,
                                           weight_decay=self.config.weight_decay)
        self.optimizer_f = torch.optim.SGD(list(self.model_f1.parameters()), lr=self.config.lr,
                                           momentum=self.config.momentum,
                                           weight_decay=self.config.weight_decay)

        if torch.cuda.is_available():
            self.generator.cuda()
            self.discriminator.cuda()

    def to_variable(self, x):
        """Convert tensor to variable."""
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)

    def to_data(self, x):
        """Convert variable to tensor."""
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data

    def reset_grad(self):
        """Zero the gradient buffers."""
        self.discriminator.zero_grad()
        self.generator.zero_grad()

    def denorm(self, x):
        """Convert range (-1, 1) to (0, 1)"""
        out = (x + 1) / 2
        return out.clamp(0, 1)

    def train(self):
        """Train generator and discriminator."""

        src_domain_lbl = Variable(torch.ones(args.batch_size).long())
        tgt_domain_lbl = Variable(torch.zeros(args.batch_size).long())

        for epoch in range(args.start_epoch, args.epochs):
            d_loss_per_epoch = 0
            c_loss_per_epoch = 0

            for ind, (source, target) in tqdm.tqdm(enumerate(train_loader)):
                src_imgs, src_lbls = Variable(source[0]), Variable(source[1])
                tgt_imgs = Variable(target[0])

                if torch.cuda.is_available():
                    src_imgs, src_lbls, tgt_imgs = src_imgs.cuda(), src_lbls.cuda(), tgt_imgs.cuda()
                    src_domain_lbl, tgt_domain_lbl = src_domain_lbl.cuda(), tgt_domain_lbl.cuda()

                # update generator and classifiers by source samples
                self.optimizer_g.zero_grad()
                self.optimizer_f.zero_grad()

                src_fet = self.model_g(src_imgs)
                tgt_fet = self.model_g(tgt_imgs)

                # for k, v in outputs.items():
                #     try:
                #         print ("%s: %s" % (k, v.size()))
                #     except AttributeError:
                #         print ("%s: %s" % (k, v))
                if "drn" in args.net:
                    src_domain_pred = self.model_d(src_fet)
                    tgt_domain_pred = self.model_d(tgt_fet)
                else:
                    src_domain_pred = self.model_d(src_fet["fm4"])
                    tgt_domain_pred = self.model_d(tgt_fet["fm4"])

                loss_d = - criterion_d(src_domain_pred, src_domain_lbl)
                loss_d -= criterion_d(tgt_domain_pred, tgt_domain_lbl)

                src_out = self.model_f(src_fet)
                loss = criterion(src_out, src_lbls)

                c_loss = loss.data[0]
                loss += loss_d
                loss.backward()
                c_loss_per_epoch += c_loss

                self.optimizer_g.step()
                self.optimizer_f.step()

                # update for classifiers
                self.optimizer_g.zero_grad()
                self.optimizer_f.zero_grad()

                src_fet = self.model_g(src_imgs)
                tgt_fet = self.model_g(tgt_imgs)
                if "drn" in args.net:
                    src_domain_pred = self.model_d(src_fet)
                    tgt_domain_pred = self.model_d(tgt_fet)
                else:
                    src_domain_pred = self.model_d(src_fet["fm4"])
                    tgt_domain_pred = self.model_d(tgt_fet["fm4"])
                loss_d = criterion_d(src_domain_pred, src_domain_lbl)
                loss_d += criterion_d(tgt_domain_pred, tgt_domain_lbl)
                loss_d.backward()
                self.optimizer_d.step()
                self.optimizer_d.zero_grad()

                d_loss = 0
                d_loss += loss_d.data[0] / args.num_k
                d_loss_per_epoch += d_loss
                if ind % 100 == 0:
                    print("iter [%d] DLoss: %.6f CLoss: %.4f" % (ind, d_loss, c_loss))

                if ind > args.max_iter:
                    break

            print("Epoch [%d] DLoss: %.4f CLoss: %.4f" % (epoch, d_loss_per_epoch, c_loss_per_epoch))
            # ploter.plot("c_loss", "train", epoch + 1, c_loss_per_epoch)
            # ploter.plot("d_loss", "train", epoch + 1, d_loss_per_epoch)
            log_value('c_loss', c_loss_per_epoch, epoch)
            log_value('d_loss', d_loss_per_epoch, epoch)
            log_value('lr', args.lr, epoch)

            if args.adjust_lr:
                args.lr = adjust_learning_rate(self.optimizer_g, args.lr, args.weight_decay, epoch, args.epochs)
                args.lr = adjust_learning_rate(self.optimizer_f, args.lr, args.weight_decay, epoch, args.epochs)

            checkpoint_fn = os.path.join(pth_dir,
                                         "%s-%s-res%s-%s.pth.tar" % (args.savename, args.net, args.res, epoch + 1))
            save_dic = {
                'epoch': epoch + 1,
                'res': args.res,
                'net': args.net,
                'args': args,
                'g_state_dict': self.model_g.state_dict(),
                'f1_state_dict': self.model_f.state_dict(),
                'd_state_dict': self.model_d.state_dict(),
                'self.optimizer_g': self.optimizer_g.state_dict(),
                'self.optimizer_f': self.optimizer_f.state_dict(),
                'self.optimizer_d': self.optimizer_d.state_dict(),
            }

            save_checkpoint(save_dic, is_best=False, filename=checkpoint_fn)

    def sample(self):

        # Load trained parameters
        g_path = os.path.join(self.model_path, 'generator-%d.pkl' % (self.num_epochs))
        d_path = os.path.join(self.model_path, 'discriminator-%d.pkl' % (self.num_epochs))
        self.generator.load_state_dict(torch.load(g_path))
        self.discriminator.load_state_dict(torch.load(d_path))
        self.generator.eval()
        self.discriminator.eval()

        # Sample the images
        noise = self.to_variable(torch.randn(self.sample_size, self.z_dim))
        fake_images = self.generator(noise)
        sample_path = os.path.join(self.sample_path, 'fake_samples-final.png')
        torchvision.utils.save_image(self.denorm(fake_images.data), sample_path, nrow=12)

        print("Saved sampled images to '%s'" % sample_path)
