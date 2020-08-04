        ax_1D[0].legend(loc="upper left")
        # ax_1D[0].set_title(r"""Posterior distributions of the variables $\lambda_1,\;\lambda_2,\;\tau$""")
        # ax_1D[0].set_xlim(320,2500)
        ax_1D[0].set_xlabel("$\lambda_1$ value", labelpad=8)
        ax_1D[0].set_ylabel("Density", labelpad=8);
        ax_1D[0].grid(which='major', linestyle=':', linewidth='0.3', color='gray')

        #
        ax_1D[1].hist(lambda_2_samples, histtype='stepfilled', bins=30, alpha=0.85,
                    label="posterior of $\lambda_2$", color="#7A68A6", density=True)
        ax_1D[1].legend(loc="upper left")
        ax_1D[1].set_xlabel("$\lambda_2$ value", labelpad=8)
        ax_1D[1].set_ylabel("Density", labelpad=8);
        ax_1D[1].grid(which='major', linestyle=':', linewidth='0.3', color='gray')

        #
        w = 1.0 / tau_samples.shape[0] * np.ones_like(tau_samples)  # weights
        ax_1D[2].hist(tau_samples, bins=num_records, alpha=1,
                    label=r"posterior of $\tau$",
                    color="#467821", edgecolor= "#467821", weights=w, linewidth='2', rwidth=0)
        ax_1D[2].set_xlabel(r"$\tau$ (in hours)", labelpad=8)
        ax_1D[2].set_ylabel("Probability", labelpad=8)
        ax_1D[2].set_xlim([25, num_records-35])
        ax_1D[2].legend(loc="upper left")
        ax_1D[2].set_xticks(np.arange(num_records)[::6])
        ax_1D[2].grid(which='major', linestyle=':', linewidth='0.3', color='gray')