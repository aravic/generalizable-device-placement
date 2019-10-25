import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr-init', type=float, default=1e-3)
parser.add_argument('--lr-dec', type=float, default=.95)
parser.add_argument('--lr-start-decay-step', type=int, default=int(1e9))
parser.add_argument('--lr-decay-steps', type=int, default=100)
parser.add_argument('--lr-min', type=float, default=1e-3)
parser.add_argument('--lr-dec-approach', type=str, default='exponential')

parser.add_argument('--ent-dec-init', type=float, default=1.)
parser.add_argument('--ent-dec', type=float, default=.95)
parser.add_argument('--ent-start-dec-step', type=int, default=int(1e9))
parser.add_argument('--ent-dec-steps', type=int, default=100)
parser.add_argument('--ent-dec-min', type=float, default=float(0.0))
parser.add_argument('--ent-dec-approach', type=str, default='linear')
parser.add_argument('--ent-dec-lin-steps', type=int, default=0)

parser.add_argument('--optimizer-type', type=str, default='adam')

# parser.add_argument('--bl-dec', type=float, default=.95)
# parser.add_argument('--bl-init', type=float, default=0)


parser.add_argument('--eps-init', type=float, default=.0)
parser.add_argument('--eps-dec-rate', type=float, default=.95)
parser.add_argument('--start-eps-dec-step', type=float, default=int(1e9))
parser.add_argument('--eps-dec-steps', type=int, default=int(1e9))
parser.add_argument('--stop-eps-dec-step', type=int, default=int(1e9))

parser.add_argument('--no-grad-clip', action='store_true', dest='no_grad_clip')

parser.add_argument('--tanhc-init', type=float, default=None)
parser.add_argument('--tanhc-dec-steps', type=int, default=None)
parser.add_argument('--tanhc-start-dec-step', type=int, default=0)
parser.add_argument('--tanhc-max', type=float, default=None)

args, unknown = parser.parse_known_args()

globals().update(args.__dict__)
