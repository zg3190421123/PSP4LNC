import torch.nn.functional as F
import torchmetrics
import torch

def GpromptEva_new(loader, gnn_head,head_prompt,gnn_body,body_prompt,gnn_tail,tail_prompt,w_h,w_b,w_t,center_embedding, num_class, device):
    head_prompt.eval()
    body_prompt.eval()
    tail_prompt.eval()

    accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_class).to(device)
    macro_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_class, average="macro").to(device)
    auroc = torchmetrics.classification.AUROC(task="multiclass", num_classes=num_class).to(device)
    auprc = torchmetrics.classification.AveragePrecision(task="multiclass", num_classes=num_class).to(device)

    accuracy.reset()
    macro_f1.reset()
    auroc.reset()
    auprc.reset()

    with torch.no_grad():
        for batch_id, batch in enumerate(loader):
            batch = batch.to(device)

            out_head = gnn_head(batch.x, batch.edge_index, batch.batch, head_prompt, 'Gprompt')
            out_body = gnn_body(batch.x, batch.edge_index, batch.batch, body_prompt, 'Gprompt')
            out_tail = gnn_tail(batch.x, batch.edge_index, batch.batch, tail_prompt, 'Gprompt')

            head_similarity_matrix = F.cosine_similarity(out_head.unsqueeze(1), center_embedding.unsqueeze(0), dim=-1)
            body_similarity_matrix = F.cosine_similarity(out_body.unsqueeze(1), center_embedding.unsqueeze(0), dim=-1)
            tail_similarity_matrix = F.cosine_similarity(out_tail.unsqueeze(1), center_embedding.unsqueeze(0), dim=-1)

            weighted_pred = w_h * head_similarity_matrix + w_b * body_similarity_matrix + w_t *tail_similarity_matrix
            pred = weighted_pred.argmax(dim=1)
            acc = accuracy(pred, batch.y)
            ma_f1 = macro_f1(pred, batch.y)
            roc = auroc(weighted_pred, batch.y)
            prc = auprc(weighted_pred, batch.y)
            if len(loader) > 20:
                print("Batch {}/{} Acc: {:.4f} | Macro-F1: {:.4f}| AUROC: {:.4f}| AUPRC: {:.4f}".format(batch_id, len(loader), acc.item(), ma_f1.item(), roc.item(), prc.item()))

    acc = accuracy.compute()
    ma_f1 = macro_f1.compute()
    roc = auroc.compute()
    prc = auprc.compute()

    return acc.item(), ma_f1.item(), roc.item(), prc.item()





