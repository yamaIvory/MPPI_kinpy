import numpy as np
from dynamics import Dynamics

class MPPIController:
    def __init__(self, urdf_path):
        # 1. Kinpy 기반의 Dynamics 객체 생성
        self.dyn = Dynamics(urdf_path)
        
        # [수정 포인트] Pinocchio 의존성(model.nq) 제거
        # Kinpy Chain에서 관절 개수(nq)를 직접 가져옵니다.
        self.nq = len(self.dyn.chain.get_joint_parameter_names())
        
        # 2. MPPI 파라미터 설정 (튜닝 가능)
        self.horizon = 20       # 얼마나 먼 미래까지 내다볼지 (Step 수)
        self.n_samples = 50     # 한 번에 몇 개의 시나리오를 뿌릴지
        self.noise_sigma = 0.1  # 탐색 노이즈 크기 (작으면 정밀, 크면 과감)
        self.lambda_ = 1.0      # 온도 파라미터
        
        # 제어 입력 범위 (관절 속도 rad/s)
        self.u_min = -1.0
        self.u_max = 1.0

    def compute_cost(self, q_curr, target_pos):
        """
        비용 함수: 목표 지점과의 거리 + (선택) 제어 입력 크기
        """
        # 현재 로봇의 끝단 위치 계산 (Kinpy Forward Kinematics)
        # Dynamics 클래스의 step 메서드를 활용해도 되지만, 여기선 위치만 필요하므로 직접 호출
        trans = self.dyn.chain.forward_kinematics(q_curr)
        ee_pos = trans.pos
        
        # 목표와의 거리 (유클리드 거리 제곱)
        dist_cost = 10.0 * np.sum((ee_pos - target_pos)**2)
        
        return dist_cost

    def compute_action(self, q_curr, target_pos):
        """
        MPPI 알고리즘의 핵심 루프
        """
        # 1. 노이즈 생성 (랜덤한 관절 속도 명령들)
        # shape: (n_samples, horizon, nq)
        noise = np.random.normal(0, self.noise_sigma, (self.n_samples, self.horizon, self.nq))
        
        costs = np.zeros(self.n_samples)
        
        # 2. 시뮬레이션 (Rollout)
        # 여러 개의 평행 우주(Sample)를 시뮬레이션 돌려봄
        for k in range(self.n_samples):
            q_sim = q_curr.copy()
            
            for t in range(self.horizon):
                # 노이즈가 섞인 제어 입력 (u = noise)
                u = noise[k, t, :]
                
                # dynamics.step을 이용해 다음 상태 예측
                # (q, pos, rot, vel)을 반환하지만, 여기선 q_next만 필요
                q_next, _, _, _ = self.dyn.step(q_sim, u)
                
                # 상태 업데이트
                q_sim = q_next
                
                # 비용 누적 (목표물에 가까울수록 비용이 낮음)
                costs[k] += self.compute_cost(q_sim, target_pos)
        
        # 3. 비용 기반 가중치 계산 (Softmax와 유사)
        # 비용이 낮은(좋은) 궤적일수록 높은 가중치를 가짐
        min_cost = np.min(costs)
        weights = np.exp(-1.0/self.lambda_ * (costs - min_cost))
        weights /= np.sum(weights) + 1e-10 # 0 나누기 방지
        
        # 4. 최적의 제어 입력 계산 (가중 평균)
        # 시간 t=0 에서의 최적 입력만 구함 (MPC 방식)
        best_u = np.zeros(self.nq)
        for k in range(self.n_samples):
            best_u += weights[k] * noise[k, 0, :]
            
        return best_u