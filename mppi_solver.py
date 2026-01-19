import numpy as np
from dynamics import Dynamics

class MPPIController:
    def __init__(self, urdf_path):
        self.dyn = Dynamics(urdf_path)
        
        # 로봇 관절 개수 (Kinpy 체인에서 가져옴)
        self.nq = len(self.dyn.chain.get_joint_parameter_names())

        # ---- MPPI Hyperparameters (다이어트 적용됨) ----
        self.K = 20             # [수정] 샘플 개수 대폭 감소 (500 -> 50)
        self.N = 10             # [수정] 미래 예측 단계 감소 (30 -> 15)
        self.dt = 0.1
        self.dyn.dt = self.dt
        self.lambda_ = 0.6      
        self.alpha = 0.3        

        # Cost weights (가중치)
        self.w_pos = 150.0
        self.w_rot = 20.0
        self.w_pos_terminal = 300.0
        self.w_rot_terminal = 50.0
        self.w_vel = 0.1

        # Noise covariance
        self.sigma = np.array([1.0]*3 + [0.5]*3)
        self.sigma_sq = self.sigma**2

        # Nominal control sequence
        self.U = np.zeros((self.N, 6))

        # 환경 설정
        self.desk_height = 0.05

    # ---------------------------------------------------
    # State cost
    # ---------------------------------------------------
    def state_cost(self, ee_pos, ee_rot, P_goal, R_goal):
        pos_err = np.linalg.norm(ee_pos - P_goal)
        rot_err = 3.0 - np.trace(np.dot(R_goal.T, ee_rot))
        cost = (self.w_pos * pos_err**2) + (self.w_rot * rot_err)
        return cost
   
    # ---------------------------------------------------
    # Terminal cost
    # ---------------------------------------------------
    def terminal_cost(self, ee_pos, ee_rot, P_goal, R_goal):
        pos_err = np.linalg.norm(ee_pos - P_goal)
        rot_err = 3.0 - np.trace(np.dot(R_goal.T, ee_rot))
        return self.w_pos_terminal*pos_err**2 + self.w_rot_terminal*rot_err

    # --------------------------------------------------- 
    # Height Cost
    # ---------------------------------------------------
    def get_height_cost(self, ee_pos):
        z_pos = ee_pos[2]
        if z_pos < self.desk_height:
            return 1e9 
        return 0.0

    # --------------------------------------------------- 
    # Joint Limit Cost
    # ---------------------------------------------------
    def get_joint_limit_cost(self, q):
        margin = 0.05
        if np.any(q < self.dyn.q_min) or np.any(q > self.dyn.q_max):
            return 1e9 
        
        diff_lower = (self.dyn.q_min + margin) - q
        cost_lower = np.sum(np.maximum(0, diff_lower)**2)
        diff_upper = q - (self.dyn.q_max - margin)
        cost_upper = np.sum(np.maximum(0, diff_upper)**2)

        w_limit = 100.0
        return w_limit * (cost_lower + cost_upper)

    # ---------------------------------------------------
    # MPPI Main Routine
    # ---------------------------------------------------
    # [중요] 여기에 R_goal=None이 추가되어야 에러가 안 납니다!
    def compute_action(self, q_curr, P_goal, R_goal=None):
        if R_goal is None:
            R_goal = np.eye(3)

        # 1. 노이즈 생성
        noise = np.random.normal(loc=0.0, scale=self.sigma, size=(self.K, self.N, 6))
        costs = np.zeros(self.K)

        # 2. Rollouts
        for k in range(self.K):
            q_sim = q_curr.copy()
            S = 0.0 

            for t in range(self.N):
                u_nom = self.U[t]
                du = noise[k, t]
                u = u_nom + du 

                # 속도 클리핑
                v_limit = 0.5
                w_limit = 2.0
                u[:3] = np.clip(u[:3], -v_limit, v_limit)
                u[3:] = np.clip(u[3:], -w_limit, w_limit)

                # Dynamics Step
                q_next, ee_pos, ee_rot, _ = self.dyn.step(q_sim, u)

                # 비용 계산
                h_cost = self.get_height_cost(ee_pos)
                l_cost = self.get_joint_limit_cost(q_next)
                s_cost = self.state_cost(ee_pos, ee_rot, P_goal, R_goal)
                ctrl_cost = self.w_vel * np.sum(u**2)

                S += (s_cost + ctrl_cost) * self.dt + h_cost + l_cost

                # 충돌 시 조기 종료
                if h_cost > 1e8 or l_cost > 1e8:
                    S += 1e9 * (self.N - t) 
                    break 

                q_sim = q_next

            if S < 1e8: 
                S += self.terminal_cost(ee_pos, ee_rot, P_goal, R_goal)

            costs[k] = S

        # 3. 가중치 계산
        beta = np.min(costs)
        weights = np.exp(-1.0/self.lambda_ * (costs - beta))
        weights /= np.sum(weights) + 1e-10

        # 4. 업데이트
        delta_U = np.sum(weights[:, None, None] * noise, axis=0)
        U_new = self.U + delta_U
        self.U = (1 - self.alpha) * self.U + self.alpha * U_new

        # 최종 출력 제한
        v_limit = 0.5
        w_limit = 2.0
        self.U[:, :3] = np.clip(self.U[:, :3], -v_limit, v_limit)
        self.U[:, 3:] = np.clip(self.U[:, 3:], -w_limit, w_limit)

        # 5. Shift
        u_opt = self.U[0].copy()
        self.U = np.roll(self.U, -1, axis=0)
        self.U[-1] = np.zeros(6) 

        return u_opt
