import re
import argparse
import matplotlib.pyplot as plt
import numpy as np


def running_mean(x, n):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[n:] - cumsum[:-n]) / float(n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_files', type=str, nargs='+')
    parser.add_argument('--plot_psnr', default=True, action='store_true')
    parser.add_argument('--plot_ssim', default=False, action='store_true')
    parser.add_argument('--plot_lpips', default=False, action='store_true')
    parser.add_argument('--plot_rgb_loss', default=False, action='store_true')
    parser.add_argument('--plot_ssim_loss', default=False, action='store_true')
    # parser.add_argument('--plot_rgb_loss2', default=False, action='store_true')
    # parser.add_argument('--plot_ssim_loss2', default=False, action='store_true')
    parser.add_argument('--plot_train', default=True, action='store_true')
    parser.add_argument('--plot_dynamic', default=False, action='store_true')
    parser.add_argument('--plot_human', default=False, action='store_true')
    parser.add_argument('--plot_vehicle', default=False, action='store_true')
    parser.add_argument('--plot_nonsky', default=False, action='store_true')
    parser.add_argument('--window_size', type=int, default=1)
    args = parser.parse_args()

    # define the names of the models you want to plot and the longest episodes you want to show
    # models = ['CPDDeform@60v3','CPDDeform@60v2','CPDDeform@40', 'CPDDeform@100']
    models = ['Omnirev1', 'CPDv1','CPDv2','CPDv3', 'CPDv4']
    max_episodes = 30000

    ax1 = ax2 = ax3 = ax4 = ax5 = ax6= None
    ax1_legends = []
    ax2_legends = []
    ax3_legends = []
    ax4_legends = []
    ax5_legends = []
    ax6_legends = []
    # ax5_legends = []

    # Full Image
    full_image_pattern = r"I\d{8} \d{2}:\d{2}:\d{2} .+?\] \s*Full Image\s+(?P<metric>PSNR|SSIM|LPIPS):\s+(?P<value>\d+\.\d+)"

    # Non-Sky
    non_sky_pattern    = r"I\d{8} \d{2}:\d{2}:\d{2} .+?\] \s*Non-Sky\s+(?P<metric>PSNR|SSIM):\s+(?P<value>\d+\.\d+)"

    # Dynamic-Only
    dynamic_only_pattern = r"I\d{8} \d{2}:\d{2}:\d{2} .+?\] \s*Dynamic-Only\s+(?P<metric>PSNR|SSIM):\s+(?P<value>\d+\.\d+)"

    # Human-Only
    human_only_pattern   = r"I\d{8} \d{2}:\d{2}:\d{2} .+?\] \s*Human-Only\s+(?P<metric>PSNR|SSIM):\s+(?P<value>\d+\.\d+)"

    # Vehicle-Only
    vehicle_only_pattern = r"I\d{8} \d{2}:\d{2}:\d{2} .+?\] \s*Vehicle-Only\s+(?P<metric>PSNR|SSIM):\s+(?P<value>\d+\.\d+)"

    # 
    general_pattern = r"I\d{8} \d{2}:\d{2}:\d{2} .+?\] \s*(?P<category>Full Image|Non-Sky|Dynamic-Only|Human-Only|Vehicle-Only)\s+(?P<metric>PSNR|SSIM|LPIPS):\s+(?P<value>\d+\.\d+)"

    progress_pattern = r"I\d{8} \d{2}:\d{2}:\d{2} .+?\]\s*\[\s*(?P<current>\d+)/\s*(?P<total>\d+)\]"
    # r"I\d{8} \d{2}:\d{2}:\d{2} .+?\]\s*\[\s*(?P<current>\d+)/\s*(?P<total>\d+)\]"

    combined_pattern = r"I\d{8} \d{2}:\d{2}:\d{2} .+?\]\s*\[\s*(?P<current>\d+)/\s*(?P<total>\d+)\].*?losses/rgb_loss:\s+(?P<rgb_loss>\d+\.\d+)\s+\((?P<rgb_loss_avg>\d+\.\d+)\).*?losses/ssim_loss:\s+(?P<ssim_loss>\d+\.\d+)\s+\((?P<ssim_loss_avg>\d+\.\d+)\)"


    results_pattern = r"I(?P<date>\d{8})\s+(?P<time>\d{2}:\d{2}:\d{2})\s+.+?\s+(?P<file>[\w_]+\.py):(?P<line>\d+)\]\s+Frame\s+(?P<frame>\d+):\s+PSNR\s+(?P<psnr>\d+\.\d+),\s+SSIM\s+(?P<ssim>\d+\.\d+)"

    for i, log_file in enumerate(args.log_files):
        with open(log_file, 'r') as file:
            log = file.read()

        # train_pattern = r"TRAIN in episode (?P<episode>\d+) has success rate: (?P<sr>[0-1].\d+), " \
        #                 r"collision rate: (?P<cr>[0-1].\d+), nav time: (?P<time>\d+.\d+), " \
        #                 r"total reward: (?P<reward>[-+]?\d+.\d+)"
        train_episode = []
        train_psnr    = []
        train_ssim    = []
        train_lpips   = []
        train_dynamic_psnr = []
        train_dynamic_ssim = []
        train_nonsky_psnr  = []
        train_nonsky_ssim  = []
        train_human_psnr   = []
        train_human_ssim   = []
        train_vehicle_psnr = []
        train_vehicle_ssim = []
        rgb_loss = []
        rgb_loss2 = []
        ssim_loss = []
        ssim_loss2 = []
        # train_reward = []
        # for r in re.findall(progress_pattern, log):
        #     train_episode.append(int(r[0]))

        for r in re.findall(combined_pattern, log):
            # combined_match = re.search(combined_pattern, log, re.DOTALL)
            train_episode.append(int(r[0]))
            rgb_loss.append(float(r[2]))
            rgb_loss2.append(float(r[3]))
            ssim_loss.append(float(r[4]))
            ssim_loss2.append(float(r[5]))

        for r in re.findall(full_image_pattern, log):
            if r[0] == "PSNR":
                train_psnr.append(float(r[1]))
            if r[0] == "SSIM":
                train_ssim.append(float(r[1]))
            if r[0] == "LPIPS":
                train_lpips.append(float(r[1]))

        for r in re.findall(dynamic_only_pattern, log):
            if r[0] == "PSNR":
                train_dynamic_psnr.append(float(r[1]))
            if r[0] == "SSIM":
                train_dynamic_ssim.append(float(r[1]))

        for r in re.findall(non_sky_pattern, log):
            if r[0] == "PSNR":
                train_nonsky_psnr.append(float(r[1]))
            if r[0] == "SSIM":
                train_nonsky_ssim.append(float(r[1]))

        for r in re.findall(human_only_pattern, log):
            if r[0] == "PSNR":
                train_human_psnr.append(float(r[1]))
            if r[0] == "SSIM":
                train_human_ssim.append(float(r[1]))

        for r in re.findall(vehicle_only_pattern, log):
            if r[0] == "PSNR":
                train_vehicle_psnr.append(float(r[1]))
            if r[0] == "SSIM":
                train_vehicle_ssim.append(float(r[1]))
  
        train_episode = train_episode[:max_episodes]
        rgb_loss    = rgb_loss[:max_episodes]
        rgb_loss2   = rgb_loss2[:max_episodes]
        ssim_loss   = ssim_loss[:max_episodes]
        ssim_loss2  = ssim_loss2[:max_episodes]
        train_psnr  = train_psnr[:max_episodes]
        train_ssim  = train_ssim[:max_episodes]
        train_lpips = train_lpips[:max_episodes]
        train_dynamic_psnr = train_dynamic_psnr[:max_episodes]
        train_dynamic_ssim = train_dynamic_ssim[:max_episodes]
        train_nonsky_psnr  = train_nonsky_psnr[:max_episodes]
        train_nonsky_ssim  = train_nonsky_ssim[:max_episodes]
        train_human_psnr   = train_human_psnr[:max_episodes]
        train_human_ssim   = train_human_ssim[:max_episodes]
        train_vehicle_psnr = train_vehicle_psnr[:max_episodes]
        train_vehicle_ssim = train_vehicle_ssim[:max_episodes]
        # train_reward = train_reward[:max_episodes]

        # # smooth training plot
        rgb_loss_smooth = running_mean(rgb_loss, args.window_size)
        rgb_loss2_smooth = running_mean(rgb_loss2, args.window_size)
        ssim_loss_smooth = running_mean(ssim_loss, args.window_size)
        ssim_loss2_smooth = running_mean(ssim_loss2, args.window_size)
        train_psnr_smooth = running_mean(train_psnr, args.window_size)
        train_ssim_smooth = running_mean(train_ssim, args.window_size)
        train_lpips_smooth = running_mean(train_lpips, args.window_size)
        train_dynamic_psnr_smooth = running_mean(train_dynamic_psnr, args.window_size)
        train_dynamic_ssim_smooth = running_mean(train_dynamic_ssim, args.window_size)
        train_nonsky_psnr_smooth = running_mean(train_nonsky_psnr, args.window_size)
        train_nonsky_ssim_smooth = running_mean(train_nonsky_ssim, args.window_size)
        train_human_psnr_smooth = running_mean(train_human_psnr, args.window_size)
        train_human_ssim_smooth = running_mean(train_human_ssim, args.window_size)
        train_vehicle_psnr_smooth = running_mean(train_vehicle_psnr, args.window_size)
        train_vehicle_ssim_smooth = running_mean(train_vehicle_ssim, args.window_size)
        # train_reward_smooth = running_mean(train_reward, args.window_size)

        # plot sr
        if args.plot_psnr:
            if ax1 is None:
                _, ax1 = plt.subplots()
            if args.plot_train:
                ax1.plot(range(len(train_psnr_smooth)), train_psnr_smooth)
                ax1_legends.append(models[i])
                if args.plot_dynamic:
                    ax1.plot(range(len(train_dynamic_psnr_smooth)), train_dynamic_psnr_smooth)
                    ax1_legends.append(models[i]+"_dynamic")
                if args.plot_nonsky:
                    ax1.plot(range(len(train_nonsky_psnr_smooth)), train_nonsky_psnr_smooth)
                    ax1_legends.append(models[i]+"_non-sky")
                if args.plot_human:
                    ax1.plot(range(len(train_human_psnr_smooth)), train_human_psnr_smooth, linestyle='-.')
                    ax1_legends.append(models[i]+"_human")
                if args.plot_vehicle:
                    ax1.plot(range(len(train_vehicle_psnr_smooth)), train_vehicle_psnr_smooth, linestyle='--')
                    ax1_legends.append(models[i]+"_vehicle")
                
            ax1.legend(ax1_legends)
            ax1.set_xlabel('Episodes')
            ax1.set_ylabel('PSNR')
            ax1.set_title('PSNR')

        # plot time
        if args.plot_ssim:
            if ax2 is None:
                _, ax2 = plt.subplots()
            if args.plot_train:
                ax2.plot(range(len(train_ssim_smooth)), train_ssim_smooth)
                if args.plot_dynamic:
                    ax2.plot(range(len(train_dynamic_ssim_smooth)), train_dynamic_ssim_smooth)
                    ax2_legends.append(models[i]+"_dynamic")
                if args.plot_nonsky:
                    ax2.plot(range(len(train_nonsky_ssim_smooth)), train_nonsky_ssim_smooth)
                    ax2_legends.append(models[i]+"_non-sky")
                if args.plot_human:
                    ax2.plot(range(len(train_human_ssim_smooth)), train_human_ssim_smooth, linestyle='-.')
                    ax2_legends.append(models[i]+"_human")
                if args.plot_vehicle:
                    ax2.plot(range(len(train_vehicle_ssim_smooth)), train_vehicle_ssim_smooth,linestyle='--')
                    ax2_legends.append(models[i]+"_vehicle")
                ax2_legends.append(models[i])

            ax2.legend(ax2_legends)
            ax2.set_xlabel('Episodes')
            ax2.set_ylabel('SSIM')
            ax2.set_title("SSIM")

        # plot cr
        if args.plot_lpips:
            if ax3 is None:
                _, ax3 = plt.subplots()
            if args.plot_train:
                ax3.plot(range(len(train_lpips_smooth)), train_lpips_smooth)
                ax3_legends.append(models[i])
   

            ax3.legend(ax3_legends)
            ax3.set_xlabel('Episodes')
            ax3.set_ylabel('LPIPS')
            ax3.set_title('LPIPS')

        # plot reward
        if args.plot_rgb_loss:
            if ax4 is None:
                _, ax4 = plt.subplots()
            if args.plot_train:
                ax4.plot(range(len(rgb_loss_smooth)), rgb_loss_smooth)
                ax4.plot(range(len(rgb_loss2_smooth)), rgb_loss2_smooth)
                ax4_legends.append(models[i]+"_rgb_loss")
                ax4_legends.append(models[i]+"_rgb_loss2")
     

            ax4.legend(ax4_legends)
            ax4.set_xlabel('Episodes')
            ax4.set_ylabel('RGB_LOSS')
            ax4.set_title('RGB_LOSS')
        
        if args.plot_ssim_loss:
            if ax5 is None:
                _, ax5 = plt.subplots()
            if args.plot_train:
                ax5.plot(range(len(ssim_loss_smooth)), ssim_loss_smooth)
                ax5.plot(range(len(ssim_loss2_smooth)), ssim_loss2_smooth)
                ax5_legends.append(models[i]+"_ssim_loss")
                ax5_legends.append(models[i]+"_ssim_loss2")
          

            ax5.legend(ax4_legends)
            ax5.set_xlabel('Episodes')
            ax5.set_ylabel('SSIM_LOSS')
            ax5.set_title('SSIM_LOSS')

        
    plt.grid(True) 
    plt.show()


if __name__ == '__main__':
    main()