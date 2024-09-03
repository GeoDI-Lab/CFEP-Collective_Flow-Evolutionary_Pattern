import sys,os
import networkx as nx
import matplotlib.pyplot as plt 
import random
import pandas as pd
import numpy as np
import seaborn as sns
from numpy import random
import scipy.stats
import geopandas as gpd
from shapely.geometry import Point, LineString
import pickle
from sklearn.linear_model import LinearRegression

    
def FG_dict2list(FG_dict):
    FG_list=dict()
    for i in set(FG_dict.values()):
        FG_list[i]=[]
    for key,value in FG_dict.items():
        FG_list[value].append(key)
    return list(FG_list.values())


# merge FGmunity result to fishnet. Set result_FG to null if Corresponding FGmunity only have one node.
def merge_CFEP_to_fishnet(result_CFEP,fishnet_gpd):
    FGmunity_number=set(result_CFEP.values())
    FG_node_list=dict()
    for FG in FGmunity_number:
        FG_node_list[FG]=[]
    for node in result_CFEP:
        FG_node_list[result_CFEP[node]].append(node)

    one_node_FG=[]
    for FG in FG_node_list:
        if len(FG_node_list[FG])==1:
            one_node_FG.append(FG)

    df_FG=pd.DataFrame.from_dict(result_CFEP,orient='index',columns=['result_FG'])
    df_FG['cell_index']=df_FG.index
    df_FG['cell_index']=pd.to_numeric(df_FG['cell_index'],errors='coerce')
    fishnet_gpd=fishnet_gpd.merge(df_FG, on='cell_index', how = 'outer')

    for index,content in fishnet_gpd.iterrows():
        if content['result_FG'] in one_node_FG:
            fishnet_gpd.iloc[index,-1]=np.nan
    return fishnet_gpd,FG_node_list

def merge_contribution_to_fishnet(contri_FG,fishnet_gpd):

    df_FG=pd.DataFrame.from_dict(contri_FG,orient='index',columns=['contri_Q','avg_Q','avg_change_rate'])
    df_FG['result_FG']=df_FG.index
    df_FG['result_FG']=pd.to_numeric(df_FG['result_FG'],errors='coerce')
    fishnet_gpd=fishnet_gpd.merge(df_FG, on='result_FG', how = 'outer')

    return fishnet_gpd

def get_each_contri(G_d1,G_d2,FG_dict):
    total_weight_d1=0
    for edge in G_d1.edges().data('weight'):
        total_weight_d1+=edge[2]

    total_weight_d2=0
    for edge in G_d2.edges().data('weight'):
        total_weight_d2+=edge[2]
    
    contri_Q=dict()
    counter=0
    edge_q=[]
    for FG,node_list in FG_dict.items():
        if len(node_list)==1:
            continue
        else:
            sub_G_d1=G_d1.subgraph(node_list)
            sub_G_d2=G_d2.subgraph(node_list)
            sub_Q=0
            q=[]
            change_rate_list=[]
            for edge_d2 in sub_G_d2.edges().data('weight'):

                degree_d1=G_d1.degree(weight='weight')[edge_d2[0]]*G_d1.degree(weight='weight')[edge_d2[1]]
                degree_d2=G_d2.degree(weight='weight')[edge_d2[0]]*G_d2.degree(weight='weight')[edge_d2[1]]

                if edge_d2[2]==0 or degree_d2==0:
                    sub_Q+=0

                else:
                    q_value=sub_G_d1[edge_d2[0]][edge_d2[1]]['weight']/sub_G_d2[edge_d2[0]][edge_d2[1]]['weight']-(degree_d1/degree_d2)/(total_weight_d1/total_weight_d2)
                    sub_Q+=q_value
                    q.append(q_value)
                    change_rate_list.append(sub_G_d1[edge_d2[0]][edge_d2[1]]['weight']/sub_G_d2[edge_d2[0]][edge_d2[1]]['weight'])
                    
                
            edge_q.append(q)
            if len(sub_G_d1.edges())==0:
                continue
            
            if len(change_rate_list)==0:
                contri_Q[FG]=[sub_Q,sub_Q/len(sub_G_d1.edges()),-1]
            else:
                contri_Q[FG]=[sub_Q,sub_Q/len(sub_G_d1.edges()),sum(change_rate_list)/len(change_rate_list)]

    return contri_Q,edge_q

def Q_contribution(fishnet_3km,result_max_end_day,G_weekend,G_weekday):
    fishnet_copy=fishnet_3km.copy()
    max_end_day,max_FG=merge_CFEP_to_fishnet(result_max_end_day,fishnet_copy)
    max_contri,edge_q=get_each_contri(G_weekend,G_weekday,max_FG)
    max_end_day_Q=merge_contribution_to_fishnet(max_contri,max_end_day)
    return max_end_day_Q,edge_q

    

def real_change_rate(target_FGmunity,G_p2,G_p1):
    intra_edges=list(G_p2.subgraph(target_FGmunity).edges())

    inter_edges=[]
    rest_nodes=set(G_p2.nodes())-set(target_FGmunity)

    for node_in_group in target_FGmunity:
        for node_outside_group in rest_nodes:
            if G_p2.has_edge(node_in_group,node_outside_group):
                inter_edges.append((node_in_group,node_outside_group))

    intra_rates=[]
    for edge in intra_edges:
        if G_p1[edge[0]][edge[1]]['weight']!=0:
            intra_rates.append(G_p2[edge[0]][edge[1]]['weight']/G_p1[edge[0]][edge[1]]['weight'])
    inter_rates=[]
    for edge in inter_edges:
        if G_p1[edge[0]][edge[1]]['weight']!=0:
            inter_rates.append(G_p2[edge[0]][edge[1]]['weight']/G_p1[edge[0]][edge[1]]['weight'])
    
    return intra_rates,inter_rates


def shuffle_change_rate(intra_rates,inter_rates,diff_mean_real,iter_num=999):
    intra_rates_and_inter_rates=intra_rates+inter_rates
    #inter 0, intra 1
    intra_rates_label=[1 for i in intra_rates]
    inter_rates_label=[0 for i in inter_rates]
    intra_label_inter_label=intra_rates_label+inter_rates_label
    diff_mean=[]

    for i in range(iter_num):
        random_rate=intra_label_inter_label.copy()
        random.shuffle(random_rate)
        new_intra_rates=[]
        new_inter_rates=[]
        for index,label in enumerate(random_rate):
            if label==1:
                new_intra_rates.append(intra_rates_and_inter_rates[index])
            else:
                new_inter_rates.append(intra_rates_and_inter_rates[index])
        mean_intra=sum(new_intra_rates)/len(new_intra_rates)
        mean_inter=sum(new_inter_rates)/len(new_inter_rates)
        diff_mean.append(mean_intra-mean_inter)
    return diff_mean

def stat_mean_p_value(FG_dict,G_p2,G_p1,pattern):
    FG_list=FG_dict2list(FG_dict)
    FG_more_than_1_node=[FG for FG in FG_list if len(FG)>1]

    labels=[]
    if pattern=='Positive Co-stable':
        labels=['$F^{p,-}_{'+str(i+1)+'}$' for i in range(len(FG_more_than_1_node))]
    if pattern=='Co-increasing':
        labels=['$F^{p,+}_{'+str(i+1)+'}$' for i in range(len(FG_more_than_1_node))]
    if pattern=='Negative Co-stable':
        labels=['$F^{n,+}_{'+str(i+1)+'}$' for i in range(len(FG_more_than_1_node))]
    if pattern=='Co-decreasing':
        labels=['$F^{n,-}_{'+str(i+1)+'}$' for i in range(len(FG_more_than_1_node))]

    pseudo_p_value_list=[]
    intra_edge_num=[]
    inter_edge_num=[]
    intra_change_mean=[]
    inter_change_mean=[]
    mean_diff=[]
    for FG in FG_more_than_1_node:
        iter_num=999
        intra_rates,inter_rates=real_change_rate(FG,G_p2,G_p1)
        intra_edge_num.append(len(intra_rates))
        inter_edge_num.append(len(inter_rates))
        if len(inter_rates)==0 or len(intra_rates)==0:
            pseudo_p_value_list.append(None)
            intra_change_mean.append(None)
            inter_change_mean.append(None)
            mean_diff.append(None)
            continue

        intra_change_mean.append(sum(intra_rates)/len(intra_rates))
        inter_change_mean.append(sum(inter_rates)/len(inter_rates))
        diff_mean_real=sum(intra_rates)/len(intra_rates)-sum(inter_rates)/len(inter_rates)
        mean_diff.append(diff_mean_real)
        diff_mean=shuffle_change_rate(intra_rates,inter_rates,diff_mean_real,iter_num)
        count_random_larger_than_real=0
        for diff in diff_mean:
            if pattern in ['Positive Co-stable','Co-decreasing']:
                if diff<=diff_mean_real:
                    count_random_larger_than_real+=1
            else:
                if diff>=diff_mean_real:
                    count_random_larger_than_real+=1
        pseudo_p_value=(count_random_larger_than_real+1)/(iter_num+1)
        pseudo_p_value_list.append(pseudo_p_value)



    stat_df=pd.DataFrame.from_dict({'Flow groups':labels,
                                    'Reshuffled Internal-flow':intra_edge_num,
                                    'Reshuffled External-flow':inter_edge_num,
                                    'Average intra-change rate':intra_change_mean,
                                    'Average inter-change rate':inter_change_mean,
                                    'Difference of edge change rate':mean_diff,
                                    'Pseudo p-value':pseudo_p_value_list})
    stat_df.set_index('Flow groups')
    return stat_df,FG_more_than_1_node

def get_flow_in_FG_weight(selected_FG,case_study_result,fishnet_3km_4326,weekday_G,weekend_G):
    interset_FG=set(selected_FG.result_FG.values)
    FG_list=dict()
    for i in interset_FG:
        FG_list[i]=[]
    for key,value in case_study_result.items():
        if value in interset_FG:
            FG_list[value].append(key)

    return flow_in_FG_weight(FG_list,fishnet_3km_4326,weekday_G,weekend_G)
    
def flow_in_FG_weight(FG_dict,fishnet_3km_4326,weekday_G,weekend_G,is_FG_dict=True):

    top_edge=[]
    FG_list=[]
    if is_FG_dict:
        FG_list=FG_dict.values()
    else:
        FG_list=FG_dict
        
    for index,FG in enumerate(FG_list):
        sub_org_graph=weekday_G.subgraph(FG)
        for edge in sub_org_graph.edges():
            change_rate=0
            if weekend_G[edge[0]][edge[1]]['weight']!=0:
                change_rate=weekday_G[edge[0]][edge[1]]['weight']/weekend_G[edge[0]][edge[1]]['weight']
            else:
                change_rate=-1
            top_edge.append([edge[0],edge[1],index,change_rate,weekday_G[edge[0]][edge[1]]['weight'],weekend_G[edge[0]][edge[1]]['weight']])

    top_edge_df=pd.DataFrame(top_edge,columns=['O','D','FG_index','change_rate','weekday_weight','weekend_weight'])
    
    return top_edge_df

def load_detection_result():
    file = open('results/co_in_CFEP', 'rb')
    co_in_CFEP = pickle.load(file)
    file.close()
    
    file = open('results/po_co_CFEP', 'rb')
    po_co_CFEP = pickle.load(file)
    file.close()
    
    file = open('results/co_de_CFEP', 'rb')
    co_de_CFEP = pickle.load(file)
    file.close()
    
    file = open('results/ne_co_CFEP', 'rb')
    ne_co_CFEP = pickle.load(file)
    file.close()
    return co_in_CFEP,po_co_CFEP,co_de_CFEP,ne_co_CFEP

def select_sig_flow_groups(detection,weekday_G,weekend_G,CFEP_name):
    stat_df,FG_more_than_1_node=stat_mean_p_value(detection,
                                                   weekday_G,weekend_G,
                                                   CFEP_name)
    sig_FG=stat_df[stat_df['Pseudo p-value']<0.05]
    sig_FG_node=[FG_more_than_1_node[i] for i in sig_FG.index]
    return sig_FG,sig_FG_node
    
    
def select_integrated_size(sig_FG,ascend,top_edge_number_percentage,top_change_rate):
    sig_FG_co_in=sig_FG.sort_values(by=['Reshuffled Internal-flow','Average intra-change rate'],ascending=ascend)
    
    select_num=int(top_edge_number_percentage*len(sig_FG))
    sig_FG_co_in_select=sig_FG_co_in[:select_num]
    sig_FG_co_in_select=sig_FG_co_in_select.sort_values(by=['Average intra-change rate'],ascending=ascend[1])
    sig_FG_co_in_select=sig_FG_co_in_select[:top_change_rate]
    return sig_FG_co_in_select

def extract_flow(flow_group_list,weekday,weekend,G_dist,sig_FG):
    weekend_weight=[]
    weekday_weight=[]
    FG_avg_weekend_weight=[]
    FG_avg_weekday_weight=[]
    FG_diameter=[]
    FG_avg_dist=[]
    FG_std_dist=[]
    for FG in flow_group_list:
        sub_weekend=weekend.subgraph(FG)
        sub_weekday=weekday.subgraph(FG)
        sub_G_dist=G_dist.subgraph(FG)
        FG_weekend_weight=[]
        FG_weekday_weight=[]
        sub_G_dist_weight=[]
        for edge in sub_weekend.edges():
            weekday_weight.append(sub_weekday[edge[0]][edge[1]]['weight'])
            weekend_weight.append(sub_weekend[edge[0]][edge[1]]['weight'])
            FG_weekday_weight.append(sub_weekday[edge[0]][edge[1]]['weight'])
            FG_weekend_weight.append(sub_weekend[edge[0]][edge[1]]['weight'])
        
        for edge in sub_G_dist.edges():
            sub_G_dist_weight.append(sub_G_dist[edge[0]][edge[1]]['weight'])
            
        FG_std_dist.append(np.std(sub_G_dist_weight))
        FG_avg_dist.append(np.mean(sub_G_dist_weight))
        FG_diameter.append(nx.diameter(sub_G_dist,weight='weight'))
        
        if len(FG_weekend_weight)!=0 and len(FG_weekday_weight)!=0:
            FG_avg_weekend_weight.append(sum(FG_weekend_weight)/len(FG_weekend_weight))
            FG_avg_weekday_weight.append(sum(FG_weekday_weight)/len(FG_weekday_weight))
        else:
            FG_avg_weekend_weight.append(-1)
            FG_avg_weekday_weight.append(-1)
        
            
    df_flow=pd.DataFrame.from_dict({'weekday_weight':weekday_weight,
                                    'weekend_weight':weekend_weight})
    
    sig_FG['FG_avg_flow_weekday_weight']=FG_avg_weekday_weight
    sig_FG['FG_avg_flow_weekend_weight']=FG_avg_weekend_weight
    sig_FG['Diameter']=FG_diameter
    sig_FG['avg_distance']=FG_avg_dist
    sig_FG['std_distance']=FG_std_dist
    sig_FG['dif_day_end']=sig_FG.FG_avg_flow_weekday_weight-sig_FG.FG_avg_flow_weekend_weight

    return df_flow,sig_FG

def four_CFEP_summary(sig_FG_co_in,sig_FG_po_co,sig_FG_co_de,sig_FG_ne_co):
   
    CFEP_name=['Co-increasing','Positive co-stable','Co-decreasing','Negative co-stable']

    co_in_FG_num=[len(sig_FG_co_in),
                   sig_FG_co_in['Reshuffled Internal-flow'].mean(),
                   sig_FG_co_in['Reshuffled Internal-flow'].std(),
                   sig_FG_co_in['Average intra-change rate'].mean(),
                   sig_FG_co_in['Average intra-change rate'].std(),
                   sig_FG_co_in['Diameter'].mean()]
    co_de_FG_num=[len(sig_FG_co_de),
                   sig_FG_co_de['Reshuffled Internal-flow'].mean(),
                   sig_FG_co_de['Reshuffled Internal-flow'].std(),
                   sig_FG_co_de['Average intra-change rate'].mean(),
                   sig_FG_co_de['Average intra-change rate'].std(),
                   sig_FG_co_de['Diameter'].mean()]
    po_co_FG_num=[len(sig_FG_po_co),
                   sig_FG_po_co['Reshuffled Internal-flow'].mean(),
                   sig_FG_po_co['Reshuffled Internal-flow'].std(),
                   sig_FG_po_co['Average intra-change rate'].mean(),
                   sig_FG_po_co['Average intra-change rate'].std(),
                   sig_FG_po_co['Diameter'].mean()]
    ne_co_FG_num=[len(sig_FG_ne_co),
                   sig_FG_ne_co['Reshuffled Internal-flow'].mean(),
                   sig_FG_ne_co['Reshuffled Internal-flow'].std(),
                   sig_FG_ne_co['Average intra-change rate'].mean(),
                   sig_FG_ne_co['Average intra-change rate'].std(),
                   sig_FG_ne_co['Diameter'].mean()]
    summary_info=pd.DataFrame.from_dict({CFEP_name[0]: co_in_FG_num,
                           CFEP_name[1]: po_co_FG_num,
                           CFEP_name[2]: co_de_FG_num,
                           CFEP_name[3]: ne_co_FG_num,})
    summary_info.index=['Number of significant flow groups',
                        'Average edge number',
                        'Std of edge number',
                        'Average flow group change rate',
                        'Std of flow group change rate',
                        'Average diameter(KM)',
                        ]
    summary_info=summary_info.T
    return summary_info

def linear_fit_avg_weight_delta_without_log(data,function_name,x_pre_max,x_pre_min):
    prediction_list=[]
    for i in range(len(data)):
        df=data[i][(data[i].FG_avg_flow_weekend_weight!=0) & (data[i].FG_avg_flow_weekday_weight!=0)]
        df=df.sort_values(by=['FG_avg_flow_weekend_weight'])
        weekend_weight=df.FG_avg_flow_weekend_weight.values
        weekend_weight=weekend_weight.reshape(-1, 1)
        weekday_weight=df.FG_avg_flow_weekday_weight.values-df.FG_avg_flow_weekend_weight.values
        weekday_weight=weekday_weight.reshape(-1, 1)
        reg = LinearRegression()
        reg.fit(weekend_weight, weekday_weight)
        a = reg.coef_[0][0]     
        b = reg.intercept_[0]
        r_squared = reg.score(weekend_weight, weekday_weight)
        print(function_name[i],'：Y = %.6fX + %.6f' % (a, b),'r_squared:',r_squared)
        x_pre=[[x_pre_min[i]],[x_pre_max[i]]]
        prediction = reg.predict(x_pre)
        prediction=(prediction).flatten()
        prediction_list.append(prediction)
    return prediction_list

    
def flow_in_FG_OD(flows_in_fg,fishnet_3km_4326):

    O_x=[]
    O_y=[]
    D_x=[]
    D_y=[]
    for index,content in flows_in_fg.iterrows():
        O_point=fishnet_3km_4326[fishnet_3km_4326.cell_index==int(content['O'])]['center']
        D_point=fishnet_3km_4326[fishnet_3km_4326.cell_index==int(content['D'])]['center']
        O_x.append(float(O_point.x))
        O_y.append(float(O_point.y))
        D_x.append(float(D_point.x))
        D_y.append(float(D_point.y))
    
    flows_in_fg['O_x']=O_x
    flows_in_fg['O_y']=O_y
    flows_in_fg['D_x']=D_x
    flows_in_fg['D_y']=D_y
    
    return flows_in_fg

def get_flows_in_top5(weekday_G, weekend_G, fishnet, sig_fg, top_5_sig_fg, sig_fg_node, detection_results):
    fishnet['center']=fishnet['geometry'].centroid
    top_5_sig_fg_node=[]
    for i in range(len(top_5_sig_fg)):
        row_number = sig_fg.index.get_loc(sig_fg[sig_fg['Flow groups'] == top_5_sig_fg['Flow groups'].values[i]].index[0])
        top_5_sig_fg_node.append(sig_fg_node[row_number])

    sig_FG_list=[detection_results[f[0]] for f in top_5_sig_fg_node]
    fishnet_cells_in_fg,_=Q_contribution(fishnet,detection_results,weekday_G,weekend_G)
    selected_cells=fishnet_cells_in_fg[fishnet_cells_in_fg['result_FG'].isin(sig_FG_list)]
    flows_in_fg=get_flow_in_FG_weight(selected_cells,
                              detection_results,
                              fishnet,
                              weekday_G,
                              weekend_G)
    avg_change_rate=[]
    for i in flows_in_fg.FG_index.unique():
        current_flow=flows_in_fg[flows_in_fg.FG_index==i]
        avg_change_rate_value=current_flow[current_flow.change_rate!=-1].change_rate.mean()
        for j in range(len(current_flow)):
            avg_change_rate.append(avg_change_rate_value)

    flows_in_fg['avg_change_rate_FG']=avg_change_rate
    flows_in_fg=flows_in_fg[flows_in_fg.change_rate != -1]
    return flow_in_FG_OD(flows_in_fg,fishnet)