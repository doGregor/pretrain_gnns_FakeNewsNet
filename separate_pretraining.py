from preprocessing import *
from models import NodeDecoder, GraphDecoder


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def pretrain_context_prediction(model_graph, model_context, data_loader, optimizer_graph, optimizer_context, criterion, epochs=20):
    model_graph.to(DEVICE)
    model_context.to(DEVICE)
    
    model_graph.train()
    model_graph.train()
    
    for epoch in range(1, epochs+1):        
        balanced_loss_accum = 0
        acc_accum = 0

        for batch_idx, batch in enumerate(data_loader):
            batch.to(DEVICE)
            
            graph_rep = model_graph(batch.x_dict, batch.edge_index_dict)
            graph_rep_vector = graph_rep['article']
            
            # cat and forward
            # graph_rep_vector = model_graph(batch.x_dict, batch.edge_index_dict, batch.batch_dict)

            context_graph = get_context_graph(batch)
            context_rep_vector = model_context(context_graph.x_dict, context_graph.edge_index_dict, context_graph.batch_dict)
            neg_context_rep_vector = torch.cat([context_rep_vector[cycle_index(graph_rep_vector.size()[0], i+1)] for i in range(1)], dim = 0)

            pred_pos = torch.sum(graph_rep_vector * context_rep_vector, dim = 1)
            pred_neg = torch.sum(graph_rep_vector * neg_context_rep_vector, dim = 1)

            loss_pos = criterion(pred_pos.double(), torch.ones(len(pred_pos)).to(pred_pos.device).double())
            loss_neg = criterion(pred_neg.double(), torch.zeros(len(pred_neg)).to(pred_neg.device).double())

            optimizer_graph.zero_grad()
            optimizer_context.zero_grad()

            loss = loss_pos + loss_neg
            loss.backward()

            optimizer_graph.step()
            optimizer_context.step()

            balanced_loss_accum += float(loss_pos.detach().cpu().item() + loss_neg.detach().cpu().item())
            acc_accum += 0.5 * (float(torch.sum(pred_pos > 0).detach().cpu().item())/len(pred_pos) + float(torch.sum(pred_neg < 0).detach().cpu().item())/len(pred_neg))
            
        print(f'epoch: {epoch:03d}, loss: {balanced_loss_accum/batch_idx:.4f}, accuracy: {acc_accum/batch_idx:.4f}')
            
    return balanced_loss_accum/batch_idx, acc_accum/batch_idx, model_graph


def pretrain_node_reconstruction(encoder_model, data_loader, encoder_optimizer, node_type='tweet', epochs=100, save_best=False, patience=5, model_name='pretrained_nodes', verbose=1, decoder_dim=64):
    assert node_type in ['tweet', 'user', None]
    
    node_decoder_model = NodeDecoder(decoder_dim, 768, node_type)
    node_decoder_model.to(DEVICE)
    optimizer_node_decoder_model = torch.optim.Adam(node_decoder_model.parameters(), lr=0.001)
    loss_fct_nodes = torch.nn.L1Loss()
    encoder_model.to(DEVICE)
    
    if save_best:
        epochs_without_improvement = 0
        best_loss = 9999999999
        
    encoder_model.train()
    node_decoder_model.train()
        
    for epoch in range(1, epochs+1):
        if save_best and epochs_without_improvement > patience:
            break

        node_loss = []
        for batch_idx, batch in enumerate(data_loader):
            batch.to(DEVICE)

            node_embeddings = encoder_model(batch.x_dict, batch.edge_index_dict, batch.batch_dict)
            if node_type is None:
                all_nodes_reconstructed = []
                all_nodes_original = []
                for n in ['tweet', 'user']:
                    nodes_reconstructed = node_decoder_model(node_embeddings)[batch[n]['mask']]
                    nodes_original = batch[n]['x_unmasked'][batch[n]['mask']]
                    all_nodes_reconstructed.append(nodes_reconstructed)
                    all_nodes_original.append(nodes_original)
                nodes_reconstructed = torch.cat(all_nodes_reconstructed, 0)
                nodes_original = torch.cat(all_nodes_original, 0)
            else:
                nodes_reconstructed = node_decoder_model(node_embeddings)[batch[node_type]['mask']]
                nodes_original = batch[node_type]['x_unmasked'][batch[node_type]['mask']]

            loss_nodes = loss_fct_nodes(nodes_reconstructed, nodes_original)
            node_loss.append(loss_nodes.item())
            
            encoder_optimizer.zero_grad()
            optimizer_node_decoder_model.zero_grad()
        
            loss_nodes.backward()

            encoder_optimizer.step()
            optimizer_node_decoder_model.step()            
            
        if save_best and sum(node_loss)/len(node_loss) < best_loss:
            torch.save(encoder_model.state_dict(), model_name + '.pth')
            best_loss = sum(node_loss)/len(node_loss)
            print('save')
            epochs_without_improvement = 0
        elif save_best:
            epochs_without_improvement += 1
        if verbose == 1:
            print(f'epoch: {epoch:03d}, loss: {sum(node_loss)/len(node_loss):.4f}')
    return encoder_model


def pretrain_graph_level(encoder_model, data_loader, encoder_optimizer, epochs=100, save_best=False, patience=5, model_name='pretrained_nodes', verbose=1, decoder_dim=64):
    graph_decoder_model = GraphDecoder(decoder_dim, 1)
    graph_decoder_model.to(DEVICE)
    optimizer_graph_decoder_model = torch.optim.Adam(graph_decoder_model.parameters(), lr=0.001)
    loss_fct_graphs = torch.nn.L1Loss()
    encoder_model.to(DEVICE)
    
    if save_best:
        best_loss = 9999999999
    
    encoder_model.train()
    graph_decoder_model.train()

    for epoch in range(1, epochs+1):
        all_loss = []
        for batch_idx, batch in enumerate(data_loader):
            batch.to(DEVICE)

            node_embeddings = encoder_model(batch.x_dict, batch.edge_index_dict, batch.batch_dict)

            graph_predictions = graph_decoder_model(node_embeddings, batch.batch_dict)
            graph_true = batch['article']['graph_target']
            graph_true = graph_true[:, None]

            loss_graphs = loss_fct_graphs(graph_predictions, graph_true)
            all_loss.append(loss_graphs.item())

            encoder_optimizer.zero_grad()
            optimizer_graph_decoder_model.zero_grad()

            loss_graphs.backward()

            encoder_optimizer.step()
            optimizer_graph_decoder_model.step()

        if save_best and sum(all_loss)/len(all_loss) < best_loss:
            torch.save(encoder_model.state_dict(), model_name+'.pth')
            best_loss = sum(all_loss)/len(all_loss)
            print('save')
        if verbose == 1:
            print('epoch:', epoch, 'loss:', sum(all_loss)/len(all_loss))
    return encoder_model
