python cartpole_with_noise.py --mode train --policy ppo 
--steps 2000000 --plan-b --no-plan-b-reward --force-field --field-strength 1.0 
--max-blocks 15 --spawn-prob 0.15 --max-episode-steps 5000 --log-interval 10000
--policy-path models/ppo_planb.pt


python cartpole_with_noise.py --mode train --policy ppo --steps 2000000 --plan-b --no-plan-b-reward --k-nearest 3 --force-field --field-strength 50 --field-pole-strength 50 --max-blocks 5 --spawn-prob 0.15 --max-episode-steps 5000 --horizontal-only --acc-min 25 --acc-max 30 --spawn-y-band-half 2 --log-interval 10000 --policy-path models/ppo.pt