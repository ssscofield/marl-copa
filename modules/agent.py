import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.attention import MHA


class Agent(nn.Module):
    def __init__(self, args):
        super(Agent, self).__init__()
        self.args = args
        dc = args.attribute_dim
        do = args.observation_dim
        de = args.entity_dim

        dh = args.agent_hidden_dim
        dh_coach = args.coach_hidden_dim if self.args.has_coach or self.args.has_coaches else 0

        n_heads = args.n_heads

        self.na = args.n_actions

        self.mha = MHA(input_dim=dh, hidden_dim=dh, n_heads=args.n_heads)

        self.personal = nn.Sequential(
            nn.Linear(dh, dh),
            nn.ReLU(),
            nn.Linear(dh, dh)
        )

        self.action_embedding = nn.Embedding(self.na, self.na)
        self.action_embedding.weight.data = torch.eye(self.na).to(args.device)
        for p in self.action_embedding.parameters():
            p.requires_grad = False

        self.agent_encode = nn.Linear(do + dc + self.na, dh)
        self.other_encode = nn.Linear(de, dh)

        self.fc1 = nn.Linear(dh, dh)
        self.rnn = nn.GRUCell(dh, dh)
        self.rnn_dh = dh

        self.q_head = nn.Sequential(
            nn.Linear(dh + dh_coach, dh),
            nn.ReLU(),
            nn.Linear(dh, self.na)
        )
        if self.args.has_coach or self.args.has_coaches:
            self.z_team = None

        if self.args.has_coaches:
            self.z_team_list = None

    def init_hidden(self, n, n_agents):
        return self.fc1.weight.new(n, n_agents, self.rnn_dh).zero_()

    def personal_hidden(self, x, m):
        h = self.mha(x, m)
        return self.personal(h)

    def tensorize(self, o, e, c, m, ms, mc_list):
        bs, n_agents = c.shape[:2]
        device = self.args.device
        o = torch.Tensor(o).to(device)
        e = torch.Tensor(e).to(device)
        c = torch.Tensor(c).to(device) - 0.5
        m = torch.Tensor(m).to(device)
        ms = torch.Tensor(ms).to(device)
        mc_list = torch.Tensor(mc_list).to(device)
        return o, e, c, m, ms, mc_list

    def tensorize_list(self, mc_list):
        mc_list_ = []
        device = self.args.device
        for mc in mc_list:
            mc_ = torch.Tensor(mc).to(device)
            mc_list_.append(mc_)
        return mc_list_

    def set_team_strategy(self, z_team):
        self.z_team = z_team

    # 对应文章中的是否需要收集这次发来的消息，indices是掩码向量
    def set_part_team_strategy(self, z_team, indices):
        prev = self.z_team
        self.z_team = z_team * indices.unsqueeze(-1) + \
                      prev * (1 - indices.unsqueeze(-1))

    def set_team_strategy_list(self, z_team_list):
        self.z_team_list = z_team_list

    def q(self, h, rnn_hidden):
        """
        h: [batch, n_agents, dz]
        """
        bs, n_agents = h.shape[:2]

        h = h.view(bs * n_agents, -1)  # 是经过一个MHA和MLP后得到的状态，还需要经过GRU才能最终得到隐藏层状态
        x = F.relu(self.fc1(h))  # 这里目前意义不明

        rnn_hidden = rnn_hidden.view(-1, self.rnn_dh)
        rnn_hidden = self.rnn(h, rnn_hidden).view(bs, n_agents, -1)  # 这里经过GRU后，才会得到最终agent的ha即隐藏层状态，用于自己q网络的输入

        if self.args.has_coach or self.args.has_coaches:
            assert self.z_team is not None, "have to form team strategy first"
            if bs == self.z_team.shape[0] * 2:  # imaginary
                z_ = self.z_team.repeat(2, 1, 1)
            else:
                z_ = self.z_team
            h_ = torch.cat([rnn_hidden, z_], -1)
        # elif self.args.has_coaches:
        #     assert self.z_team_list is not None, "要先设置team strategy list"
        #     # TODO 在这里给不同的agent设置不同的zteam向量 如果有一个agent在多个coach视角里，如何安排这个向量？
        else:
            h_ = rnn_hidden

        q = self.q_head(h_)
        return q, h

    def forward(self, o, e, c, m, ms, rnn_hidden, prev_a, a=None):
        """
        o: [batch, n_agents, observation_dim]
        e: [batch, n_others, other_dim]
        c: [batch, n_agents, attribute_dim]
        m: [batch, n_agents, n_all]
        a: [batch, n_agents]
        """
        bs, n_agents = o.shape[:2]

        # 这里和原文中X（entity-representation）即实体状态矩阵对上
        # X的前na行是智能体的状态，后续是实体entity的状态
        # 这里要把characteristic向量以及前一刻动作向量拼上，因为智能体输入包含前一时刻动作信息
        # 所以实际上完整的智能体的observation o包括：观测+特征+前一刻动作
        prev_a = self.action_embedding(prev_a)  # [batch, n_agents, na] one-hot编码动作向量
        o = torch.cat([o, c, prev_a], -1)  # [batch, n_agents, na+do+dc]

        # 为了能拼接，需要encode变成同一维度
        ha = self.agent_encode(o)  # [batch, n_agents, dh]
        he = self.other_encode(e)  # [batch, n_others, dh]

        n_others = he.shape[1]  # landmark数量
        # 这里就组成了原文中的X矩阵
        x = torch.cat([ha, he], 1)  # [batch, n_all, dh]

        h_full = None
        if not self.args.has_coach and not self.args.has_coaches:
            h_full = self.personal_hidden(x, ms)  # 没有coach时，根据全局的mask矩阵，生成隐藏层状态合集[1,6,128]

        h = self.personal_hidden(x, m)  # [batch, na, dh] 根据各自的观测mask生成各自的隐藏层状态

        qa, rnn_hidden = self.q(h, rnn_hidden)  # 这里的qa维度为[1,6,5] 代表每个智能体对应每个动作的q值

        if a is not None:
            qa = qa.gather(dim=-1, index=a.unsqueeze(-1)).squeeze(-1)

        return qa, h, h_full, rnn_hidden  # 返回的分别是每个智能体对应动作的q值、局部观测下各自的隐藏层状态（没经过gru）、全局mask下隐藏层状态、经过gru生成的最终的隐藏层状态

    def im_forward(self, o, e, c, m, ms, rnn_hidden, prev_a, mi, mo, a=None):
        """
        rnn_hidden: [2*batch, n_agents] because we have within-group/betw-group
        """
        bs, n_agents = o.shape[:2]

        prev_a = self.action_embedding(prev_a)  # [batch, n_agents, na]
        o = torch.cat([o, c, prev_a], -1)

        ha = self.agent_encode(o)  # [batch, n_agents, dh]
        he = self.other_encode(e)  # [batch, n_others, dh]
        n_others = he.shape[1]

        x = torch.cat([ha, he], 1)  # [batch, n_all, dh]
        x = torch.cat([x, x], 0)  # [batch*2, n_all, dh]

        # ma = m.sum(-1).gt(0).float() # [batch, n_agent, 1]
        # me = F.pad(ma, (0, n_others), "constant", 1)
        # mm = torch.rand_like(me).le(np.random.rand()).float()
        # mi = ma.unsqueeze(-1).bmm(mm.unsqueeze(1)) + (1-ma).unsqueeze(-1).bmm(1-mm.unsqueeze(1))
        # mo = 1 - mi
        h = self.personal_hidden(x, torch.cat([mi * m, mo * m], 0))
        fmi, fmo = mi * ms, mo * ms

        qa, rnn_hidden = self.q(h, rnn_hidden)

        if a is not None:
            a = torch.cat([a, a], 0)
            qa = qa.gather(dim=-1, index=a.unsqueeze(-1)).squeeze(-1)  # [2*batch, n_agents]

        return qa, h, fmi, fmo, rnn_hidden

    def step(self, o, e, c, m, ms, rnn_hidden, prev_a, epsilon=0.1):
        bs, n_agents = o.shape[:2]
        with torch.no_grad():
            q, _, _, rnn_hidden = self.forward(o, e, c, m, ms, rnn_hidden, prev_a)
            a = torch.argmax(q, -1).cpu().numpy()  # 根据q最大的那个作为本次选取的动作 维度就是[1,6]
            prob = np.random.rand(*a.shape)
            rand_a = np.random.randint(low=0, high=self.na, size=(bs, n_agents))
            a = a * (prob > epsilon) + rand_a * (prob <= epsilon)
            return a, rnn_hidden
