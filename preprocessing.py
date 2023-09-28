import torch


def mask_tweet_nodes_and_add_retweet_count(graph, avg_feature, mask_prob=0.15):
    rt_count = graph[('tweet', 'retweets', 'tweet')].edge_index.size()[-1]
    graph['article']['rt_count'] = torch.tensor([rt_count])
    
    graph['tweet']['x_unmasked'] = torch.clone(graph['tweet']['x'])
    mask = (torch.rand(graph['tweet']['x'].shape[0]) < mask_prob).bool()
    graph['tweet']['mask'] = mask
    graph['tweet']['x'][mask] = torch.flatten(torch.full((1,768), avg_feature))

    return graph


def mask_user_nodes_and_add_user_count(graph, avg_feature, mask_prob=0.15):
    user_count = graph['user'].x.size()[0]
    graph['article']['user_count'] = torch.tensor([user_count])
    
    graph['user']['x_unmasked'] = torch.clone(graph['user']['x'])
    mask = (torch.rand(graph['user']['x'].shape[0]) < mask_prob).bool()
    graph['user']['mask'] = mask
    graph['user']['x'][mask] = torch.flatten(torch.full((1,768), avg_feature))

    return graph


def mask_nodes(graph, avg_feature_dict, mask_prob=0.15):
    for node_type, avg_feature in avg_feature_dict.items():
        graph[node_type]['x_unmasked'] = torch.clone(graph[node_type]['x'])
        mask = (torch.rand(graph[node_type]['x'].shape[0]) < mask_prob).bool()
        graph[node_type]['mask'] = mask
        graph[node_type]['x'][mask] = torch.flatten(torch.full((1,768), avg_feature))
    return graph


def get_max_rt_count(all_graphs):
    max_rt_count = max([g[('tweet', 'retweets', 'tweet')].edge_index.size()[-1] for g in all_graphs])
    return max_rt_count


def add_rt_count(graph, max_rt_count=None):
    ''' add number of retweets as graph level pretraining target '''
    rt_count = graph[('tweet', 'retweets', 'tweet')].edge_index.size()[-1]
    if max_rt_count is not None:
        graph['article']['graph_target'] = torch.tensor([rt_count/max_rt_count])
    else:
        graph['article']['graph_target'] = torch.tensor([rt_count])
    return graph


def get_max_tweet_count(all_graphs):
    max_tweet_count = max([g[('tweet', 'cites', 'article')].edge_index.size()[-1] for g in all_graphs])
    return max_tweet_count


def add_tweet_count(graph, max_tweet_count=None):
    ''' add number of tweets as graph level pretraining target '''
    tweet_count = graph[('tweet', 'cites', 'article')].edge_index.size()[-1]
    if max_tweet_count is not None:
        graph['article']['graph_target'] = torch.tensor([tweet_count/max_tweet_count])
    else:
        graph['article']['graph_target'] = torch.tensor([tweet_count])
    return graph


def get_max_user_count(all_graphs):
    max_user_count = max([g['user'].x.size()[0] for g in all_graphs])
    return max_user_count


def add_user_count(graph, max_user_count=None):
    ''' add number of users as graph level pretraining target '''
    user_count = graph['user'].x.size()[0]
    if max_user_count is not None:
        graph['article']['graph_target'] = torch.tensor([user_count/max_user_count])
    else:
        graph['article']['graph_target'] = torch.tensor([user_count])
    return graph


def add_rt_prop(graph):
    ''' add proportion of retweets/tweets as graph level pretraining target '''
    rt_prop = graph[('tweet', 'retweets', 'tweet')].edge_index.size()[-1] / graph['tweet'].x.size()[0]
    graph['article']['graph_target'] = torch.tensor([rt_prop])
    return graph


def get_context_graph(graph):
    subgraph = graph.edge_type_subgraph([('user', 'posts', 'tweet'), ('tweet', 'retweets', 'tweet'),
                                         ('tweet', 'rev_posts', 'user'), ('tweet', 'rev_retweets', 'tweet')])
    return subgraph


def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr
